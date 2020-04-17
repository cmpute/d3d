import os
import os.path as osp
import shutil
from enum import Enum
from zipfile import ZipFile

import numpy as np

from scipy.spatial.transform import Rotation
from d3d.dataset.kitti import utils
from d3d.abstraction import ObjectTarget3D, ObjectTag, ObjectTarget3DArray, TransformSet

class KittiObjectClass(Enum):
    DontCare = 0
    Car = 1
    Van = 2
    Truck = 3
    Pedestrian = 4
    Person_sitting = 5
    Cyclist = 6
    Tram = 7
    Misc = 255

class ObjectLoader:
    """
    Load and parse odometry benchmark data into a usable format.
    Please ensure that calibration and poses are downloaded
    
    # Zip Files
    - data_object_calib.zip
    - data_object_image_2.zip
    - data_object_image_3.zip
    - data_object_label_2.zip
    - data_object_velodyne.zip

    # Unzipped Structure
    - <base_path directory>
        - training
            - calib
            - image_2
            - label_2
            - velodyne
        - testing
            - calib
            - image_2
            - velodyne
    """

    VALID_CAM_NAMES = ["cam2", "cam3"]
    VALID_LIDAR_NAMES = ["velo"]

    def __init__(self, base_path, frames=None, inzip=False, phase="training"):
        """
        :param base_path: directory containing the zip files, or the required data
        :param frames: select which frames to load, can be use with fixed train val split
        :param inzip: whether the dataset is store in original zip archives or unzipped
        :param phase: training or testing
        """
        self.base_path = base_path
        self.frames = frames
        self.inzip = inzip
        self.phase = phase

        if phase not in ['training', 'testing']:
            raise ValueError("Invalid phase tag")

        # Open zipfiles and count total number of frames
        total_count = None
        if self.inzip:
            if phase == 'training':
                # label data is necessary when training
                with ZipFile(osp.join(base_path, "data_object_label_2.zip")) as label_data:
                    total_count = sum(1 for name in label_data.namelist() if not name.endswith('/'))
            elif osp.exists(osp.join(base_path, "data_object_image_2.zip")):
                with ZipFile(osp.join(base_path, "data_object_image_2.zip")) as cam2_data:
                    total_count = total_count or sum(1 for name in cam2_data.namelist() if not name.endswith('/'))
            elif osp.exists(osp.join(base_path, "data_object_velodyne.zip")):
                with ZipFile(osp.join(base_path, "data_object_velodyne.zip")) as velo_data:
                    total_count = total_count or sum(1 for name in velo_data.namelist() if not name.endswith('/'))
        else:
            if phase == 'training':
                total_count = len(os.listdir(osp.join(base_path, phase, 'label_2')))
            elif os.path.exists(osp.join(base_path, phase, 'image_2')):
                total_count = len(os.listdir(osp.join(base_path, phase, 'image_2')))
            elif os.path.exists(osp.join(base_path, phase, 'velodyne')):
                total_count = len(os.listdir(osp.join(base_path, phase, 'velodyne')))

        # Find all the data files
        self.frames = list(self.frames or range(total_count))

        # used to store the image size
        self._image_size_cache = {}

    def __len__(self):
        return len(self.frames)

    def camera_data(self, idx, names='cam2'):
        unpack_result = False
        if names is None:
            names = ObjectLoader.VALID_CAM_NAMES
        elif isinstance(names, str):
            names = [names]
            unpack_result = True
        else: # sanity check
            for name in names:
                if name not in ObjectLoader.VALID_CAM_NAMES:
                    raise ValueError("Invalid camera name, options are " +
                        ", ".join(ObjectLoader.VALID_CAM_NAMES))
        
        outputs = []
        for name in names:
            if name == "cam2":
                folder_name = "image_2"
            elif name == "cam3":
                folder_name = "image_3"

            file_name = osp.join(self.phase, folder_name, '%06d.png' % self.frames[idx])
            if self.inzip:
                with ZipFile(osp.join(self.base_path, "data_object_%s.zip" % folder_name)) as source:
                    image = utils.load_image(source, file_name, gray=False)
            else:
                image = utils.load_image(self.base_path, file_name, gray=False)

            outputs.append(image)
            if idx not in self._image_size_cache:
                self._image_size_cache[idx] = image.size

        if unpack_result:
            return outputs[0]
        else:
            return outputs

    def lidar_data(self, idx, names='velo'):
        if isinstance(names, str):
            names = [names]
        if names != ObjectLoader.VALID_LIDAR_NAMES:
            raise "There's only one lidar in KITTI dataset"

        fname = osp.join(self.phase, 'velodyne', '%06d.bin' % idx)
        if self.inzip:
            with ZipFile(osp.join(self.base_path, "data_object_velodyne.zip")) as source:
                return utils.load_velo_scan(source, fname)
        else:
            return utils.load_velo_scan(self.base_path, fname)
        source.close()

    def calibration_data(self, idx, raw=False):
        if self.inzip:
            with ZipFile(osp.join(self.base_path, "data_object_calib.zip")) as source:
                return self._load_calib(source, idx, raw)
        else:
            return self._load_calib(self.base_path, idx, raw)

    def lidar_label(self, idx):
        if self.phase == "testing":
            raise ValueError("Testing dataset doesn't contain label data")

        fname = osp.join(self.phase, 'label_2', '%06d.txt' % self.frames[idx])
        if self.inzip:
            with ZipFile(osp.join(self.base_path, "data_object_label_2.zip")) as source:
                return self._load_label(source, fname)
        else:
            return self._load_label(self.base_path, fname)

    def lidar_objects(self, idx=None):
        '''
        Return list of converted ground truth targets. Objects labelled as `DontCare` are removed
        '''
        return self._generate_objects(self.lidar_label(idx), self.calibration_data(idx, raw=True))

    def _load_calib(self, basepath, idx, raw=False):
        data = TransformSet("velo")

        # load the calibration file
        filename = osp.join(self.phase, 'calib', '%06d.txt' % self.frames[idx])
        filedata = utils.load_calib_file(basepath, filename)

        if raw:
            return filedata

        # load image size, which could take additional time
        if idx not in self._image_size_cache:
            self.camera_data(idx) # fill image size cache
        image_size = self._image_size_cache[idx]

        # load matrics
        rect = filedata['R0_rect'].reshape(3, 3)
        velo_to_cam = filedata['Tr_velo_to_cam'].reshape(3, 4)
        for i in range(4):
            P = filedata['P%d' % i].reshape(3, 4)
            intri, offset = P[:, :3], P[:, 3]
            projection = intri.dot(rect)
            offset_cartesian = np.linalg.inv(projection).dot(offset)
            extri = np.vstack([velo_to_cam, np.array([0,0,0,1])])
            extri[:3, 3] += offset_cartesian

            frame = "cam%d" % i
            data.set_intrinsic_camera(frame, projection, image_size, rotate=False)
            data.set_extrinsic(extri, frame_to=frame)

        data.set_intrinsic_general("imu")
        data.set_extrinsic(filedata['Tr_imu_to_velo'].reshape(3, 4), frame_from="imu")

        return data

    def _load_label(self, basepath, file):
        data = []
        if isinstance(basepath, str):
            fin = open(os.path.join(basepath, file))
        else: # assume ZipFile object
            fin = basepath.open(file)

        with fin:
            for line in fin.readlines():
                if not line.strip():
                    continue

                values = [KittiObjectClass[value] if idx == 0 else float(value)
                    for idx, value in enumerate(line.split(' '))]
                data.append(values)

        return data

    def _generate_objects(self, label, calib):
        Tr = calib['Tr_velo_to_cam'].reshape(3, 4)
        RRect = Rotation.from_dcm(calib['R0_rect'].reshape(3, 3))
        HR, HT = Rotation.from_dcm(Tr[:,:3]), Tr[:,3]
        objects = ObjectTarget3DArray()
        objects.frame = "velo"

        for item in label:
            if item[0] == KittiObjectClass.DontCare:
                continue

            h,w,l = item[8:11] # height, width, length
            position = item[11:14] # x, y, z in camera coordinate
            ry = item[14] # rotation of y axis in camera coordinate
            position[1] -= h/2

            # Here we ignore R0_rect since it's basically identity
            position = np.dot(position, RRect.inv().as_matrix().T)
            position = HR.inv().as_matrix().dot(position - HT)
            orientation = HR.inv() * RRect.inv() * Rotation.from_euler('y', ry)
            orientation *= Rotation.from_euler("x", np.pi/2) # change dimension from l,h,w to l,w,h

            tag = ObjectTag(item[0], KittiObjectClass)
            target = ObjectTarget3D(position, orientation, [l,w,h], tag)
            objects.append(target)

        return objects

def print_detection_result(detections: ObjectTarget3DArray, calib:TransformSet, raw_calib: dict):
    '''
    For detection output, we need detection result list, formatted calibration object and raw calibration
    for R0_rect
    '''
    assert detections.frame == "velo"
    Tr = raw_calib['Tr_velo_to_cam'].reshape(3, 4)
    RRect = Rotation.from_dcm(raw_calib['R0_rect'].reshape(3, 3))
    HR, HT = Rotation.from_dcm(Tr[:,:3]), Tr[:,3]

    output_lines = []
    output_format = "%s 0 0 0" + " %.2f" * 12
    for box in detections:
        # calculate bounding box 2D
        uv, mask = calib.project_points_to_camera(box.corners, frame_to="cam2", frame_from="velo", remove_outlier=False)
        if np.sum(mask) < 4: continue # ignore boxes that is outside the image
        umin, vmin = np.min(uv, axis=0)
        umax, vmax = np.max(uv, axis=0)

        # calculate position in original 3D frame
        l,w,h = box.dimension
        position = RRect.as_matrix().dot(HR.as_matrix().dot(box.position) + HT)
        position[1] += h/2
        orientation = box.orientation * Rotation.from_euler("x", np.pi/2)
        orientation = RRect * HR * orientation
        yaw = orientation.as_euler("YZX")[0]
        
        output_values = (box.tag_name,)
        output_values += (umin, vmin, umax, vmax)
        output_values += (h, w, l)
        output_values += tuple(position.tolist())
        output_values += (yaw, box.tag_score)
        output_lines.append(output_format % output_values)
    
    return "\n".join(output_lines)
        

def execute_official_evaluator(exec_path, label_path, result_path, output_path, sha=None, show_output=True):
    '''
    Execute official evaluator from KITTI devkit
    '''
    import subprocess
    import tempfile

    if sha is None:
        sha = "noname"

    # create temporary directory
    temp_path = tempfile.mkdtemp()
    temp_label_path = osp.join(temp_path, "data", "object")
    temp_result_path = osp.join(temp_path, "results", sha)
    if not osp.exists(temp_label_path):
        os.makedirs(temp_label_path)
    if not osp.exists(temp_result_path):
        os.makedirs(temp_result_path)
    if not osp.exists(output_path):
        os.makedirs(output_path)

    # execute
    os.symlink(label_path, osp.join(temp_label_path, "label_2"), target_is_directory=True)
    os.symlink(result_path, osp.join(temp_result_path, "data"), target_is_directory=True)
    proc = subprocess.Popen([exec_path, sha], cwd=temp_path, stdout=None if show_output else subprocess.PIPE)
    proc.wait()

    # move result files
    for dirname in os.listdir(temp_result_path):
        if dirname == "data":
            continue

        shutil.move(osp.join(temp_result_path, dirname), output_path)

    # clean
    shutil.rmtree(temp_path)

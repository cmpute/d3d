import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union
from zipfile import ZipFile
import zipfile
from io import RawIOBase

import numpy as np
from scipy.spatial.transform import Rotation

from d3d.abstraction import (ObjectTag, ObjectTarget3D, Target3DArray,
                             TransformSet)
from d3d.dataset.base import DetectionDatasetBase, check_frames, split_trainval, expand_name
from d3d.dataset.kitti import utils
from d3d.dataset.kitti.utils import KittiObjectClass
from d3d.dataset.zip import PatchedZipFile

def load_label(basepath, file):
    '''
    Load label or result from text file in KITTI object detection label format
    '''
    data = []
    if isinstance(basepath, (str, Path)):
        fin = Path(basepath, file).open()
    else:  # assume ZipFile object
        fin = basepath.open(str(file))

    with fin:
        for line in fin.readlines():
            if not line.strip():
                continue
            if isinstance(line, bytes):
                line = line.decode()

            values = [KittiObjectClass[value] if idx == 0 else float(value)
                      for idx, value in enumerate(line.split(' '))]
            data.append(values)

    return data

def parse_label(label: list, raw_calib: dict):
    '''
    Generate object array from loaded label or result text
    '''
    Tr = raw_calib['Tr_velo_to_cam'].reshape(3, 4)
    RRect = Rotation.from_matrix(raw_calib['R0_rect'].reshape(3, 3))
    HR, HT = Rotation.from_matrix(Tr[:,:3]), Tr[:,3]
    objects = Target3DArray()
    objects.frame = "velo"

    for item in label:
        if item[0] == KittiObjectClass.DontCare:
            continue

        h,w,l = item[8:11] # height, width, length
        position = item[11:14] # x, y, z in camera coordinate
        ry = item[14] # rotation of y axis in camera coordinate
        position[1] -= h/2

        # parse object position and orientation
        position = np.dot(position, RRect.inv().as_matrix().T)
        position = HR.inv().as_matrix().dot(position - HT)
        orientation = HR.inv() * RRect.inv() * Rotation.from_euler('y', ry)
        orientation *= Rotation.from_euler("x", np.pi/2) # change dimension from l,h,w to l,w,h

        score = item[15] if len(item) == 16 else None
        tag = ObjectTag(item[0], KittiObjectClass, scores=score)
        target = ObjectTarget3D(position, orientation, [l,w,h], tag)
        objects.append(target)

    return objects

def _line_box_intersect(p0, p1, width, height):
    # p0: inlier point
    # p1: outlier point
    k = (p1[1] - p0[1]) / (p1[0] - p0[0])
    
    # find which border
    case = None
    if p1[0] < p0[0]:
        if p1[1] < p0[1]:
            k0 = p0[1] / p0[0]
            case = 2 if k > k0 else 3
        else:
            k0 = - (height - p0[1]) / p0[0]
            case = 3 if k > k0 else 0
    else:
        if p1[1] < p0[1]:
            k0 = - p0[1] / (width - p0[0])
            case = 1 if k > k0 else 2
        else:
            k0 = (height - p0[1]) / (width - p0[0])
            case = 0 if k > k0 else 1

    # do intersection
    if case == 0: # y > height border
        x = p0[0] + (height - p0[1]) / k
        y = height
    elif case == 1: # x > width border
        x = width
        y = p0[1] + (width - p0[0]) * k
    elif case == 2: # y < 0 border
        x = p1[0] + (-p1[1]) / k
        y = 0
    elif case == 3: # x < 0 border
        x = 0
        y = p1[1] + (-p1[0]) * k

    assert 0 <= x <= width, "x = %.2f" % x
    assert 0 <= y <= height, "y = %.2f" % y
    return (x, y)     

class KittiObjectLoader(DetectionDatasetBase):
    """
    Loader for KITTI object detection dataset, please organize the files into following structure
    
    * Zip Files::

        - data_object_calib.zip
        - data_object_image_2.zip
        - data_object_image_3.zip
        - data_object_label_2.zip
        - data_object_velodyne.zip

    * Unzipped Structure::

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

    Note that the 3d objects labelled as `DontCare` are removed from the result of :meth:`annotation_3dobject`.

    For description of constructor parameters, please refer to :class:`d3d.dataset.base.DetectionDatasetBase`
    """

    VALID_CAM_NAMES = ["cam2", "cam3"] # TODO: rename to camera_gray_left, etc.
    VALID_LIDAR_NAMES = ["velo"]
    VALID_OBJ_CLASSES = KittiObjectClass

    def __init__(self, base_path, inzip=False, phase="training", trainval_split=0.8, trainval_random=False):
        super().__init__(base_path, inzip=inzip, phase=phase,
                         trainval_split=trainval_split, trainval_random=trainval_random)
        self.phase_path = 'training' if phase == 'validation' else phase

        # count total number of frames
        total_count = None
        if self.inzip:
            for folder in ['image_2', 'image_3', 'velodyne', 'label_2']:
                data_zip = self.base_path / ("data_object_%s.zip" % folder)
                if data_zip.exists():
                    with ZipFile(data_zip) as data:
                        total_count = total_count or sum(1 for name in data.namelist() \
                            if name.startswith(self.phase_path) and not name.endswith('/'))
                    break
        else:
            for folder in ['image_2', 'image_3', 'velodyne', 'label_2']:
                fpath = self.base_path / self.phase_path / folder
                if fpath.exists():
                    total_count = sum(1 for _ in fpath.iterdir())
                    break
        if not total_count:
            raise ValueError("Cannot parse dataset, please check path, inzip option and file structure")

        self.frames = split_trainval(phase, total_count, trainval_split, trainval_random)
        self._image_size_cache = {} # used to store the image size

    def __len__(self):
        return len(self.frames)

    def _parse_idx(self, idx):
        if isinstance(idx, int):
            uidx = self.frames[idx]
        else:
            uidx, = idx
        return uidx

    @expand_name(VALID_CAM_NAMES)
    def camera_data(self, idx, names='cam2'):
        if names == "cam2":
            folder_name = "image_2"
        elif names == "cam3":
            folder_name = "image_3"
        uidx = self._parse_idx(idx)

        fname = Path(self.phase_path, folder_name, '%06d.png' % uidx)
        if self._return_file_path:
            return self.base_path / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / ("data_object_%s.zip" % folder_name), to_extract=fname) as source:
                image = utils.load_image(source, fname, gray=False)
        else:
            image = utils.load_image(self.base_path, fname, gray=False)

        # save image size for calibration
        if uidx not in self._image_size_cache:
            self._image_size_cache[uidx] = image.size

        return image

    @expand_name(VALID_LIDAR_NAMES)
    def lidar_data(self, idx, names='velo'):
        assert names == 'velo'
        uidx = self._parse_idx(idx)

        fname = Path(self.phase_path, 'velodyne', '%06d.bin' % uidx)
        if self._return_file_path:
            return self.base_path / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / "data_object_velodyne.zip", to_extract=fname) as source:
                return utils.load_velo_scan(source, fname)
        else:
            return utils.load_velo_scan(self.base_path, fname)

    def _load_calib(self, basepath, uidx, raw=False):
        # load the calibration file
        fname = Path(self.phase_path, 'calib', '%06d.txt' % uidx)
        filedata = utils.load_calib_file(basepath, fname)

        if raw:
            return filedata

        # load image size, which could take additional time
        if uidx not in self._image_size_cache:
            self.camera_data((uidx,)) # fill image size cache
        image_size = self._image_size_cache[uidx]

        # load matrics
        data = TransformSet("velo")
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

    def calibration_data(self, idx, raw=False):
        uidx = self._parse_idx(idx)

        fname = Path(self.phase_path, 'calib', '%06d.txt' % uidx)
        if self._return_file_path:
            return self.base_path / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / "data_object_calib.zip", to_extract=fname) as source:
                return self._load_calib(source, uidx, raw)
        else:
            return self._load_calib(self.base_path, uidx, raw)

    def annotation_3dobject(self, idx, raw=False):
        assert self.phase_path != "testing", "Testing dataset doesn't contain label data"
        uidx = self._parse_idx(idx)

        fname = Path(self.phase_path, 'label_2', '%06d.txt' % uidx)
        if self._return_file_path:
            return self.base_path / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / "data_object_label_2.zip", to_extract=fname) as source:
                label = load_label(source, fname)
        else:
            label = load_label(self.base_path, fname)

        if raw:
            return label
        return parse_label(label, self.calibration_data((uidx,), raw=True))

    def identity(self, idx):
        return (self.frames[idx],)

    def dump_detection_output(self, idx: Union[int, tuple], detections: Target3DArray, fout: Union[str, Path, RawIOBase]) -> None:
        '''
        Save the detection in KITTI output format. We need raw calibration for R0_rect
        '''
        uidx = self._parse_idx(idx)
        calib = self.calibration_data(uidx)
        raw_calib = self.calibration_data(uidx, raw=True)

        # get intrinsics
        assert detections.frame == "velo"
        Tr = raw_calib['Tr_velo_to_cam'].reshape(3, 4)
        RRect = Rotation.from_matrix(raw_calib['R0_rect'].reshape(3, 3))
        HR, HT = Rotation.from_matrix(Tr[:,:3]), Tr[:,3]

        meta = calib.intrinsics_meta['cam2']
        width, height = meta.width, meta.height

        # process detections
        output_lines = []
        output_format = "%s 0 0 0" + " %.2f" * 12
        for box in detections:
            # calculate bounding box 2D
            uv, mask, dmask = calib.project_points_to_camera(box.corners,
                frame_to="cam2", frame_from="velo", remove_outlier=False, return_dmask=True)
            if len(uv[mask]) < 1: continue # ignore boxes that is outside the image

            bdpoints = []
            pairs = [(0, 1), (2, 3), (4, 5), (6, 7), # box lines
                    (0, 4), (1, 5), (2, 6), (3, 7),
                    (0, 2), (1, 3), (4, 6), (5, 7)]
            inlier = [i in mask for i in range(len(uv))]
            for i, j in pairs:
                if not inlier[i] and not inlier[j]:
                    continue
                if i not in dmask or j not in dmask: # only calculate for points ahead
                    continue
                if not inlier[i]:
                    bdpoints.append(_line_box_intersect(uv[j], uv[i], width, height))
                if not inlier[j]:
                    bdpoints.append(_line_box_intersect(uv[i], uv[j], width, height))

            uv = np.array(uv[mask].tolist() + bdpoints)
            umin, vmin = np.min(uv, axis=0)
            umax, vmax = np.max(uv, axis=0)

            # calculate position in original 3D frame
            l,w,h = box.dimension
            position = RRect.as_matrix().dot(HR.as_matrix().dot(box.position) + HT)
            position[1] += h/2
            orientation = box.orientation * Rotation.from_euler("x", np.pi/2)
            orientation = RRect * HR * orientation
            yaw = orientation.as_euler("YZX")[0]

            output_values = (box.tag_top.name,)
            output_values += (umin, vmin, umax, vmax)
            output_values += (h, w, l)
            output_values += tuple(position.tolist())
            output_values += (yaw, box.tag_top_score)
            output_lines.append(output_format % output_values)

        content = "\n".join(output_lines)
        if isinstance(fout, (str, Path)):
            Path(fout).write_text(content)
        else:
            fout.write(content.encode())

def execute_official_evaluator(exec_path, label_path, result_path, output_path, model_name=None, show_output=True):
    '''
    Execute official evaluator from KITTI devkit

    :param label_path: path to the extracted labels of kitti dataset
    :param result_path: path to the results created by dump_detection_output
    :parma output_path: path to the output evaluation results
    :param model_name: unique name of your model. KITTI tool requires sha1 as model name, but that's not mandatory.
    :param show_output: show_output option passed to Kitti evaluator

    Note: to install prerequisites `sudo apt install gnuplot texlive-extra-utils`
    '''
    model_name = model_name or "noname"

    # create temporary directory
    temp_path = Path(tempfile.mkdtemp())
    temp_label_path = temp_path / "data" / "object"
    temp_result_path = temp_path / "results" / model_name
    temp_label_path.mkdir(parents=True, exist_ok=True)
    temp_result_path.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # execute
        (temp_label_path / "label_2").symlink_to(label_path, target_is_directory=True)
        (temp_result_path / "data").symlink_to(result_path, target_is_directory=True)
        proc = subprocess.Popen([exec_path, model_name], cwd=temp_path, stdout=None if show_output else subprocess.PIPE)
        proc.wait()

        # move result files
        for dirname in temp_result_path.iterdir():
            if dirname.name == "data":
                continue
            shutil.move(dirname, output_path)

    finally:
        # clean
        shutil.rmtree(temp_path)

def create_submission(result_path, output_file):
    '''
    Create submission file from dumped detection results (using dump_detection_output)

    :param output_file: Output zip file. If not ended with `.zip`, `.zip` will be appended
    '''
    fsubmission = Path(output_file)
    if fsubmission.suffix != ".zip":
        fsubmission = fsubmission.parent / (fsubmission.name + ".zip")
    fsubmission.parent.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(fsubmission, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in Path(result_path).iterdir():
            archive.write(file, file.name)
    print("Submission file created at", fsubmission)

def parse_detection_output():
    from argparse import ArgumentParser
    from tqdm import tqdm

    parser = ArgumentParser(description='Convert detection output to dumped binary files with d3d object array.')

    parser.add_argument('input', type=str,
        help='Directory of detection output files')
    parser.add_argument('-o', '--output', type=str,
        help='Output directory. If not provided, it will be the same as input')
    parser.add_argument('-d', '--dataset-path', type=str, dest="dspath",
        help='Path of the KITTI object dataset')
    parser.add_argument('-p', '--phase', type=str, default='training',
        choices=['training', 'testing'], help="Dataset phase, either training or testing")
    parser.add_argument('-z', '--inzip', action="store_true",
        help="Whether the dataset is in zip archives")
    args = parser.parse_args()

    loader = KittiObjectLoader(args.dspath, inzip=args.inzip, phase=args.phase, trainval_split=1)
    input_path = Path(args.input)
    output_path = Path(args.output or args.input)
    output_path.mkdir(parents=True, exist_ok=True)
    for txtpath in tqdm(input_path.iterdir(), total=sum(1 for _ in input_path.iterdir())):
        relpath = txtpath.relative_to(input_path)
        boxes = load_label(input_path, relpath)
        calib = loader.calibration_data(int(relpath.stem), raw=True)
        boxes = parse_label(boxes, calib)
        boxes.dump(output_path / relpath.with_suffix('.objs'))

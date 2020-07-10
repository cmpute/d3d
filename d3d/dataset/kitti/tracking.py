import numpy as np
from zipfile import ZipFile
from pathlib import Path
from collections import defaultdict, OrderedDict
from scipy.spatial.transform import Rotation

from d3d.abstraction import (ObjectTag, ObjectTarget3D, ObjectTarget3DArray,
                             TransformSet)
from d3d.dataset.base import TrackingDatasetBase, check_frames, split_trainval
from d3d.dataset.kitti import utils
from d3d.dataset.kitti.utils import KittiObjectClass

def parse_label(label: list, raw_calib: dict):
    '''
    Generate object array from loaded label or result text
    '''
    Tr = raw_calib['Tr_velo_cam'].reshape(3, 4)
    RRect = Rotation.from_matrix(raw_calib['R_rect'].reshape(3, 3))
    HR, HT = Rotation.from_matrix(Tr[:,:3]), Tr[:,3]
    objects = ObjectTarget3DArray()
    objects.frame = "velo"

    for item in label:
        track_id = int(item[0])
        if item[1] == KittiObjectClass.DontCare:
            continue

        h,w,l = item[9:12] # height, width, length
        position = item[12:15] # x, y, z in camera coordinate
        ry = item[15] # rotation of y axis in camera coordinate
        position[1] -= h/2

        # parse object position and orientation
        position = np.dot(position, RRect.inv().as_matrix().T)
        position = HR.inv().as_matrix().dot(position - HT)
        orientation = HR.inv() * RRect.inv() * Rotation.from_euler('y', ry)
        orientation *= Rotation.from_euler("x", np.pi/2) # change dimension from l,h,w to l,w,h

        score = item[16] if len(item) == 17 else None
        tag = ObjectTag(item[0], KittiObjectClass, scores=score)
        target = ObjectTarget3D(position, orientation, [l,w,h], tag, id=track_id)
        objects.append(target)

    return objects 

class KittiTrackingLoader(TrackingDatasetBase):
    """
    Loader for KITTI multi-object tracking dataset, please organize the filed into following structure
    
    # Zip Files
    - data_tracking_calib.zip
    - data_tracking_image_2.zip
    - data_tracking_image_3.zip
    - data_tracking_label_2.zip
    - data_tracking_velodyne.zip
    - data_tracking_oxts.zip

    # Unzipped Structure
    - <base_path directory>
        - training
            - calib
            - image_02
            - label_02
            - oxts
            - velodyne
        - testing
            - calib
            - image_02
            - oxts
            - velodyne
    """

    VALID_CAM_NAMES = ["cam2", "cam3"]
    VALID_LIDAR_NAMES = ["velo"]

    def __init__(self, base_path, inzip=False, phase="training", trainval_split=0.8, trainval_random=False, nframes=1):
        base_path = Path(base_path)
        self.base_path = base_path
        self.inzip = inzip
        self.phase = phase
        self.phase_path = 'training' if phase == 'validation' else phase
        self.nframes = nframes

        if phase not in ['training', 'validation', 'testing']:
            raise ValueError("Invalid phase tag")

        # count total number of frames
        seq_count = None
        frame_count = defaultdict(int)
        if self.inzip:
            for folder in ['image_2', 'image_3', 'velodyne']:
                data_zip = base_path / ("data_tracking_%s.zip" % folder)
                if data_zip.exists():
                    seq_count = 0
                    with ZipFile(data_zip) as data:
                        for name in data.namelist():
                            name = Path(name)
                            if len(name.parts) != 4:
                                continue

                            phase, _, seq, frame = name.parts
                            if phase != self.phase_path:
                                continue

                            seq = int(seq)
                            seq_count = max(seq+1, seq_count) # index starts from 0
                            frame_count[seq] = max(frame_count[seq]+1, int(frame[:-4]))
                    break
        else:
            for folder in ['image_02', 'image_03', 'velodyne']:
                fpath = base_path / self.phase_path / folder
                if fpath.exists():
                    seq_count = 0
                    for seq_path in fpath.iterdir():
                        seq = int(seq_path.name)
                        frame_count[seq] = sum(1 for _ in seq_path.iterdir())
                        seq_count += 1
                    break
        if not seq_count:
            raise ValueError("Cannot parse dataset, please check path, inzip option and file structure")
        total_count = sum(frame_count.values()) - nframes * seq_count
        self.frame_dict = OrderedDict(frame_count)

        self.frames = split_trainval(phase, total_count, trainval_split, trainval_random)
        self._image_size_cache = {} # used to store the image size (for each sequence)
        self._label_cache = {} # used to store parsed label data
        self._calib_cache = {} # used to store parsed calibration data

    def __len__(self):
        return len(self.frames)

    def _locate_frame(self, idx):
        # use underlying frame index
        idx = self.frames[idx]

        for k, v in self.frame_dict.items():
            if (idx + self.nframes) < v:
                return k, (idx + self.nframes)
            idx -= (v - self.nframes)
        raise ValueError("Index larger than dataset size")

    def _preload_label(self, seq_id):
        if seq_id in self._label_cache:
            return

        file_name = Path(self.phase_path, "label_02", "%04d.txt" % seq_id)
        if self.inzip:
            with ZipFile(self.base_path / "data_tracking_label_2.zip") as source:
                text = source.read(str(file_name)).decode().split('\n')
        else:
            with open(self.base_path / file_name, "r") as fin:
                text = fin.readlines()

        self._label_cache[seq_id] = defaultdict(list)
        for line in text:
            if not line.strip():
                continue

            frame_id, track_id, remain = line.split(" ", 2)
            values = [KittiObjectClass[value] if idx == 0 else float(value)
                    for idx, value in enumerate(remain.split(' '))]
            self._label_cache[seq_id][int(frame_id)].append([int(track_id)] + values)

    def _preload_calib(self, seq_id):
        if seq_id in self._calib_cache:
            return

        filename = Path(self.phase_path, 'calib', '%04d.txt' % seq_id)
        if self.inzip:
            with ZipFile(self.base_path / "data_tracking_calib.zip") as source:    
                self._calib_cache[seq_id] = utils.load_calib_file(source, filename)
        else:
            self._calib_cache[seq_id] = utils.load_calib_file(self.base_path, filename)

    def camera_data(self, idx, names='cam2'):
        if isinstance(idx, int):
            seq_id, frame_id = self._locate_frame(idx)
        else:
            seq_id, frame_id = idx
        unpack_result, names = check_frames(names, self.VALID_CAM_NAMES)
        
        outputs = []
        for name in names:
            if name == "cam2":
                folder_name = "image_02"
                if self.inzip:
                    source = ZipFile(self.base_path / "data_tracking_image_2.zip")
            elif name == "cam3":
                folder_name = "image_03"
                if self.inzip:
                    source = ZipFile(self.base_path / "data_tracking_image_3.zip")

            data_seq = []
            for fidx in range(frame_id - self.nframes, frame_id+1):
                file_name = Path(self.phase_path, folder_name, "%04d" % seq_id, '%06d.png' % fidx)
                if self.inzip:
                    image = utils.load_image(source, file_name, gray=False)
                else:
                    image = utils.load_image(self.base_path, file_name, gray=False)
                data_seq.append(image)

            outputs.append(data_seq)
            if self.inzip:
                source.close()

        # save image size for calibration
        if seq_id not in self._image_size_cache:
            self._image_size_cache[seq_id] = image.size

        if unpack_result:
            return outputs[0]
        else:
            return outputs

    def lidar_data(self, idx, names='velo', concat=True):
        if isinstance(idx, int):
            seq_id, frame_id = self._locate_frame(idx)
        else:
            seq_id, frame_id = idx

        if isinstance(names, str):
            names = [names]
            unpack_result = True
        else:
            unpack_result = False
        if names != self.VALID_LIDAR_NAMES:
            raise ValueError("There's only one lidar in KITTI dataset")

        data_seq = []
        if self.inzip:
            source = ZipFile(self.base_path / "data_tracking_velodyne.zip")
        for fidx in range(frame_id - self.nframes, frame_id+1):
            fname = Path(self.phase_path, 'velodyne', "%04d" % seq_id, '%06d.bin' % fidx)
            if self.inzip:
                data_seq.append(utils.load_velo_scan(source, fname))
            else:
                data_seq.append(utils.load_velo_scan(self.base_path, fname))

        if self.inzip:
            source.close()
        if unpack_result:
            return data_seq
        else:
            return [data_seq]

    def calibration_data(self, idx, raw=False):
        if isinstance(idx, int):
            seq_id, _ = self._locate_frame(idx)
        else:
            seq_id, _ = idx

        return self._load_calib(seq_id, raw)

    def lidar_objects(self, idx, raw=False):
        '''
        Return list of converted ground truth targets in lidar frame.

        Note that objects labelled as `DontCare` are removed
        '''
        if self.phase_path == "testing":
            raise ValueError("Testing dataset doesn't contain label data")
        if isinstance(idx, int):
            seq_id, frame_id = self._locate_frame(idx)
        else:
            seq_id, frame_id = idx

        self._preload_label(seq_id)
        if raw:
            return [self._label_cache[seq_id][fid] for fid in range(frame_id - self.nframes, frame_id+1)]

        self._preload_label(seq_id)
        return [parse_label(self._label_cache[seq_id][fid], self._calib_cache[seq_id])
               for fid in range(frame_id - self.nframes, frame_id+1)]

    def identity(self, idx):
        '''
        For KITTI this method just return the phase and index
        '''
        seq, frame = self._locate_frame(idx)
        return self.phase_path, seq, frame

    def _load_calib(self, seq, raw=False):
        # load the calibration file data
        self._preload_calib(seq)
        filedata = self._calib_cache[seq]
        if raw:
            return filedata

        # load image size, which could take additional time
        if seq not in self._image_size_cache:
            self.camera_data((seq, self.nframes)) # load first n frames to fill image size cache
        image_size = self._image_size_cache[seq]

        # load matrics
        data = TransformSet("velo")
        rect = filedata['R_rect'].reshape(3, 3)
        velo_to_cam = filedata['Tr_velo_cam'].reshape(3, 4)
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
        data.set_extrinsic(filedata['Tr_imu_velo'].reshape(3, 4), frame_from="imu")

        return data

import numpy as np
from zipfile import ZipFile
from pathlib import Path
from collections import defaultdict, OrderedDict
from scipy.spatial.transform import Rotation

from d3d.abstraction import (ObjectTag, ObjectTarget3D, Target3DArray,
                             TransformSet, EgoPose)
from d3d.dataset.base import TrackingDatasetBase, check_frames, split_trainval
from d3d.dataset.kitti import utils
from d3d.dataset.kitti.utils import KittiObjectClass, OxtData
from d3d.dataset.zip import PatchedZipFile

def parse_label(label: list, raw_calib: dict) -> Target3DArray:
    '''
    Generate object array from loaded label or result text
    '''
    Tr = raw_calib['Tr_velo_cam'].reshape(3, 4)
    RRect = Rotation.from_matrix(raw_calib['R_rect'].reshape(3, 3))
    HR, HT = Rotation.from_matrix(Tr[:,:3]), Tr[:,3]
    objects = Target3DArray()
    objects.frame = "velo"

    for item in label:
        track_id = int(item[0]) # add 1 to prevent 0 tid
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
        tag = ObjectTag(item[1], KittiObjectClass, scores=score)
        target = ObjectTarget3D(position, orientation, [l,w,h], tag, tid=track_id)
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

    # TODO: add option to split trainval dataset by sequence instead of overall index
    def __init__(self, base_path, inzip=False, phase="training", trainval_split=0.8, trainval_random=False, nframes=1):
        self.base_path = Path(base_path)
        self.inzip = inzip
        self.phase = phase
        self.phase_path = 'training' if phase == 'validation' else phase
        self.nframes = nframes

        if phase not in ['training', 'validation', 'testing']:
            raise ValueError("Invalid phase tag")

        # count total number of frames
        frame_count = defaultdict(int)
        if self.inzip:
            for folder in ['image_2', 'image_3', 'velodyne']:
                data_zip = self.base_path / ("data_tracking_%s.zip" % folder)
                if data_zip.exists():
                    with ZipFile(data_zip) as data:
                        for name in data.namelist():
                            name = Path(name)
                            if len(name.parts) != 4: # skip directories
                                continue

                            phase, _, seq, frame = name.parts
                            if phase != self.phase_path:
                                continue

                            seq = int(seq)
                            frame_count[seq] = max(frame_count[seq],
                                int(frame[:-4]) + 1) # idx starts from 0
                    break
        else:
            for folder in ['image_02', 'image_03', 'velodyne']:
                fpath = self.base_path / self.phase_path / folder
                if fpath.exists():
                    for seq_path in fpath.iterdir():
                        seq = int(seq_path.name)
                        frame_count[seq] = sum(1 for _ in seq_path.iterdir())
                    break

        if not len(frame_count):
            raise ValueError("Cannot parse dataset, please check path, inzip option and file structure")
        self.frame_dict = OrderedDict(frame_count)

        total_count = sum(frame_count.values()) - nframes * len(frame_count)
        self.frames = split_trainval(phase, total_count, trainval_split, trainval_random)
        self._image_size_cache = {} # used to store the image size (for each sequence)
        self._label_cache = {} # used to store parsed label data
        self._calib_cache = {} # used to store parsed calibration data
        self._pose_cache = {} # used to store parsed pose data

    def __len__(self):
        return len(self.frames)

    @property
    def sequence_ids(self):
        return list(self.frame_dict.keys())

    @property
    def sequence_sizes(self):
        return dict(self.frame_dict)

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
            with PatchedZipFile(self.base_path / "data_tracking_label_2.zip", to_extract=file_name) as source:
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

        file_name = Path(self.phase_path, 'calib', '%04d.txt' % seq_id)
        if self.inzip:
            with PatchedZipFile(self.base_path / "data_tracking_calib.zip", to_extract=file_name) as source:    
                self._calib_cache[seq_id] = utils.load_calib_file(source, file_name)
        else:
            self._calib_cache[seq_id] = utils.load_calib_file(self.base_path, file_name)

    def _preload_oxts(self, seq_id):
        if seq_id in self._pose_cache:
            return

        file_name = Path(self.phase_path, 'oxts', '%04d.txt' % seq_id)
        if self.inzip:
            with PatchedZipFile(self.base_path / "data_tracking_oxts.zip", to_extract=file_name) as source:
                text = source.read(str(file_name)).decode().split('\n')
        else:
            with open(self.base_path / file_name, "r") as fin:
                text = fin.readlines()

        self._pose_cache[seq_id] = []
        for line in text:
            line = line.strip()
            if not line:
                continue

            values = list(map(float, line.strip().split(' ')))
            values[-5:] = list(map(int, values[-5:]))
            self._pose_cache[seq_id].append(OxtData(*values))

    def camera_data(self, idx, names='cam2'):
        if isinstance(idx, int):
            seq_id, frame_idx = self._locate_frame(idx)
        else:
            seq_id, frame_idx= idx
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
            for fidx in range(frame_idx - self.nframes, frame_idx+1):
                file_name = Path(self.phase_path, folder_name, "%04d" % seq_id, '%06d.png' % fidx)
                if self.inzip:
                    image = utils.load_image(source, file_name, gray=False)
                else:
                    image = utils.load_image(self.base_path, file_name, gray=False)
                data_seq.append(image)
            if self.nframes == 0: # unpack if only 1 frame
                data_seq = data_seq[0]

            outputs.append(data_seq)
            if self.inzip:
                source.close()

        # save image size for calibration
        if seq_id not in self._image_size_cache:
            self._image_size_cache[seq_id] = image.size

        return outputs[0] if unpack_result else outputs

    def lidar_data(self, idx, names='velo', concat=True):
        if isinstance(idx, int):
            seq_id, frame_idx = self._locate_frame(idx)
        else:
            seq_id, frame_idx = idx
        
        # This is the problem in KITTI dataset itself
        if seq_id == 1 and frame_idx in range(177, 181):
            raise ValueError("There is missing data in KITTI tracking dataset at seq 1, frame 177-180!")

        if isinstance(names, str):
            names = [names]
        if names != self.VALID_LIDAR_NAMES:
            raise ValueError("There's only one lidar in KITTI dataset")

        data_seq = []
        if self.inzip:
            source = ZipFile(self.base_path / "data_tracking_velodyne.zip")
        for fidx in range(frame_idx - self.nframes, frame_idx+1):
            fname = Path(self.phase_path, 'velodyne', "%04d" % seq_id, '%06d.bin' % fidx)
            if self.inzip:
                data_seq.append(utils.load_velo_scan(source, fname))
            else:
                data_seq.append(utils.load_velo_scan(self.base_path, fname))
        if self.nframes == 0: # unpack if only 1 frame
            data_seq = data_seq[0]

        if self.inzip:
            source.close()
        return data_seq

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
            seq_id, frame_idx = self._locate_frame(idx)
        else:
            seq_id, frame_idx = idx

        self._preload_label(seq_id)
        labels = [self._label_cache[seq_id][fid] for fid in range(frame_idx - self.nframes, frame_idx+1)]

        if self.nframes == 0:
            if raw: return labels[0]

            self._preload_calib(seq_id)
            return parse_label(labels[0], self._calib_cache[seq_id])
        else:
            if raw: return labels

            self._preload_calib(seq_id)
            return [parse_label(l, self._calib_cache[seq_id]) for l in labels]

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

    def pose(self, idx, raw=False):
        if isinstance(idx, int):
            seq_id, frame_idx = self._locate_frame(idx)
        else:
            seq_id, frame_idx = idx

        self._preload_oxts(seq_id)

        if self.nframes == 0:
            poses = self._pose_cache[seq_id][frame_idx]

            if raw: return poses
            return utils.parse_pose_from_oxt(poses)
        else:
            poses = [self._pose_cache[seq_id][fid] for fid in range(frame_idx - self.nframes, frame_idx+1)]

            if raw: return poses
            return [utils.parse_pose_from_oxt(p) for p in poses]

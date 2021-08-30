from collections import defaultdict
from logging.config import valid_ident
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from scipy.spatial.transform import Rotation
from addict import Dict as edict

from d3d.abstraction import EgoPose, TransformSet
from d3d.dataset.base import DatasetBase, expand_idx, expand_idx_name, split_trainval_seq
from d3d.dataset.kitti import utils
from d3d.dataset.kitti.utils import SemanticKittiClass
from d3d.dataset.zip import PatchedZipFile
from sortedcontainers import SortedDict


class KittiOdometryLoader(DatasetBase):
    """
    Loader for KITTI odometry detection dataset, please organize the files into following structure
    
    * Zip Files::
        - data_odometry_calib.zip [required]
        - data_odometry_color.zip
        - data_odometry_gray.zip
        - data_odometry_velodyne.zip
        - data_odometry_poses.zip [required]
        - data_odometry_labels.zip (SemanticKitti)

    * Unzipped Structure::

        - <base_path directory>
            - dataset
                - poses
                    - 00.txt
                    - ..
                - sequences
                    - 00
                        - image_0
                        - image_1
                        - image_2
                        - image_3
                        - velodyne_points
                        - calib.txt
                        - times.txt

    This dataset loader also supports SemanticKITTI dataset extension.
    """

    VALID_CAM_NAMES = ["cam2", "cam3"]
    VALID_LIDAR_NAMES = ["velo"]
    VALID_PTS_CLASSES = SemanticKittiClass

    def __init__(self, base_path, inzip=True, phase="training",
                 trainval_split=0.8, trainval_random=False, trainval_byseq=False, nframes=0):
        super().__init__(base_path, inzip=inzip, phase=phase,
                         trainval_split=trainval_split, trainval_random=trainval_random)

        # count total number of frames
        frame_count = defaultdict(int)
        if self.inzip:
            for folder in ['gray', 'color', 'velodyne', 'labels']:
                data_zip = self.base_path / ("data_odometry_%s.zip" % folder)
                if data_zip.exists():
                    with ZipFile(data_zip) as data:
                        for name in data.namelist():
                            name = Path(name)
                            if len(name.parts) < 5: # select data files
                                continue

                            _, _, seq, _, frame = name.parts

                            seq = int(seq)
                            frame_count[seq] = max(frame_count[seq],
                                int(name.stem) + 1) # idx starts from 0
                    break
        else:
            fpath = self.base_path / "dataset" / "sequences"
            if fpath.exists():
                for seq_path in fpath.iterdir():
                    seq = int(seq_path.name)
                    for folder in ['image_2', 'image_3', 'velodyne']:
                        frame_count[seq] = sum(1 for _ in seq_path.iterdir())
                    break

        if not len(frame_count):
            raise ValueError("Cannot parse dataset or empty dataset, please check path, inzip option and file structure")
        
        if phase in ["training", "validation"]:
            self.frame_dict = SortedDict({k: v for k, v in frame_count.items() if k <= 10})
        elif phase == "testing":
            self.frame_dict = SortedDict({k: v for k, v in frame_count.items() if k >= 11})
        else:
            raise ValueError("Incorrect phase argument!")
        self.frames = split_trainval_seq(phase, self.frame_dict, trainval_split, trainval_random, trainval_byseq)
        self.nframes = nframes
        self._image_size_cache = {} # used to store the image size (for each sequence)
        self._pose_cache = {} # used to store parsed pose data
        self._calib_cache = {} # used to store parsed calibration data
        self._timestamp_cache = {}

    def _locate_frame(self, idx):
        idx = self.frames[idx] # use underlying frame index

        for k, v in self.frame_dict.items():
            if idx < (v - self.nframes):
                return k, idx
            idx -= (v - self.nframes)
        raise KeyError("Index larger than dataset size")

    def __len__(self):
        return len(self.frames)

    @property
    def sequence_ids(self):
        return list(self.frame_dict.keys())

    @property
    def sequence_sizes(self):
        return dict(self.frame_dict)

    def _preload_calib(self, seq_id):
        if seq_id in self._calib_cache:
            return

        file_name = Path('dataset', 'sequences', '%02d' % seq_id, 'calib.txt')
        if self.inzip:
            with PatchedZipFile(self.base_path / "data_odometry_calib.zip", to_extract=file_name) as source:    
                self._calib_cache[seq_id] = utils.load_calib_file(source, file_name)
        else:
            self._calib_cache[seq_id] = utils.load_calib_file(self.base_path, file_name)

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
        velo_to_cam = filedata['Tr'].reshape(3, 4)
        for i in range(4):
            P = filedata['P%d' % i].reshape(3, 4)
            projection, offset = P[:, :3], P[:, 3]
            offset_cartesian = np.linalg.inv(projection).dot(offset)
            extri = np.vstack([velo_to_cam, np.array([0,0,0,1])])
            extri[:3, 3] += offset_cartesian

            frame = "cam%d" % i
            data.set_intrinsic_camera(frame, projection, image_size, rotate=False)
            data.set_extrinsic(extri, frame_to=frame)

        return data

    def calibration_data(self, idx, raw=False):
        assert not self._return_file_path, "The calibration is not stored in single file!"
        if isinstance(idx, int):
            seq_id, _ = self._locate_frame(idx)
        else:
            seq_id, _ = idx

        return self._load_calib(seq_id, raw)

    def _preload_poses(self, seq_id):
        # TODO: also support loading from SemanticKITTI archive
        if seq_id in self._pose_cache:
            return

        file_name = Path('dataset', 'poses', '%02d.txt' % seq_id)
        if self.inzip:
            with PatchedZipFile(self.base_path / "data_odometry_poses.zip", to_extract=file_name) as source:
                text = source.read(str(file_name)).decode().split('\n')
        else:
            with open(self.base_path / file_name, "r") as fin:
                text = fin.readlines()

        self._pose_cache[seq_id] = []
        for line in text:
            line = line.strip()
            if not line:
                continue

            values = np.array(list(map(float, line.strip().split(' '))))
            self._pose_cache[seq_id].append(values.reshape(3, 4))

    @expand_idx_name(VALID_CAM_NAMES)
    def camera_data(self, idx, names='cam2'):
        seq_id, frame_idx = idx

        _folders = {
            "cam0": ("image_0", "data_odometry_grey.zip", True),
            "cam1": ("image_1", "data_odometry_grey.zip", True),
            "cam2": ("image_2", "data_odometry_color.zip", False),
            "cam3": ("image_3", "data_odometry_color.zip", False)
        }
        folder_name, zip_name, grey = _folders[names]

        fname = Path("dataset", "sequences", "%02d" % seq_id, folder_name, '%06d.png' % frame_idx)
        if self._return_file_path:
            return self.base_path / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / zip_name, to_extract=fname) as source:
                image = utils.load_image(source, fname, gray=grey)
        else:
            image = utils.load_image(self.base_path, fname, gray=grey)

        # save image size for calibration
        if seq_id not in self._image_size_cache:
            self._image_size_cache[seq_id] = image.size

        return image

    @expand_idx_name(VALID_LIDAR_NAMES)
    def lidar_data(self, idx, names='velo', formatted=False):
        seq_id, frame_idx = idx
        assert names == 'velo'

        fname = Path('dataset', 'sequences', "%02d" % seq_id, 'velodyne', '%06d.bin' % frame_idx)
        if self._return_file_path:
            return self.base_path / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / "data_odometry_velodyne.zip", to_extract=fname) as source:
                return utils.load_velo_scan(source, fname, formatted=formatted)
        else:
            return utils.load_velo_scan(self.base_path, fname, formatted=formatted)

    @expand_idx
    def pose(self, idx, raw=False):
        seq_id, frame_idx = idx

        self._preload_poses(seq_id)
        rt = self._pose_cache[seq_id][frame_idx]
        if raw:
            return rt

        r = Rotation.from_matrix(rt[:3, :3])
        return EgoPose(r.inv().as_matrix().dot(rt[:3, 3]), r)

    @property
    def pose_name(self):
        return 'cam0'

    @expand_idx
    def identity(self, idx):
        return idx

    @expand_idx
    def identity_in_raw(self, idx):
        '''
        This function will return the corresponding identity in the KITTI raw dataset.
        '''
        seq_map = {
             0: "2011_10_03_drive_0027",
             1: "2011_10_03_drive_0042",
             2: "2011_10_03_drive_0034",
             3: "2011_09_26_drive_0067",
             4: "2011_09_30_drive_0016",
             5: "2011_09_30_drive_0018",
             6: "2011_09_30_drive_0020",
             7: "2011_09_30_drive_0027",
             8: "2011_09_30_drive_0028",
             9: "2011_09_30_drive_0033",
            10: "2011_09_30_drive_0034",
        }

        seq_id, frame_id = idx
        if seq_id not in seq_map:
            raise ValueError("Sequence mapping is not available for testing data!")
        if seq_id == 8:
            frame_id += 1100
        return seq_map[seq_id] + "_sync", frame_id

    @expand_idx_name(VALID_LIDAR_NAMES)
    def annotation_3dpoints(self, idx, names='velo'):
        seq_id, frame_idx = idx
        assert names == 'velo'

        fname = Path('dataset', 'sequences', "%02d" % seq_id, 'labels', '%06d.label' % frame_idx)
        if self._return_file_path:
            return self.base_path / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / "data_odometry_labels.zip", to_extract=fname) as ar:
                buffer = ar.read(str(fname))
        else:
            buffer = (self.base_path / fname).read_bytes()
        label = np.frombuffer(buffer, dtype='u4')

        return edict(semantic=label)

    def _preload_timestamp(self, seq_id):
        if seq_id in self._timestamp_cache:
            return

        fname = Path('dataset', 'sequences', '%02d' % seq_id, 'times.txt')

        if self.inzip:
            with PatchedZipFile(self.base_path / "data_odometry_calib.zip", to_extract=fname) as data:
                timelist = utils.load_timestamps(data, fname).astype(int) // 1000
        else:
            timelist = utils.load_timestamps(self.base_path, fname).astype(int) // 1000
        self._timestamp_cache[seq_id] = timelist

    @expand_idx
    def timestamp(self, idx, names="velo"):
        del names
        assert not self._return_file_path, "The timestamp is not stored in single file!"
        seq_id, frame_idx = idx
        self._preload_timestamp(seq_id)
        return self._timestamp_cache[seq_id][frame_idx]

if __name__ == "__main__":
    loader = KittiOdometryLoader("/mnt/storage8t/datasets/kitti", inzip=True)
    print(loader.pose(0))
    print(loader.camera_data(0))
    print(loader.lidar_data(0).shape)
    print(loader.calibration_data(0))
    print(loader.identity_in_raw(0))
    print(loader.timestamp(0))
    print(loader.annotation_3dpoints(0).semantic.shape)

    # from d3d.dataset.kitti.raw import KittiRawLoader
    # rloader = KittiRawLoader("/mnt/storage8t/datasets/kitti", inzip=True)
    # print(rloader.calibration_data(loader.identity_in_raw(0)))
    
    # print("Pose @ 100:", loader.pose(0))
    # print("Pose @ 100:", loader.pose(100))
    # p0 = rloader.pose(loader.identity_in_raw(0))
    # p1 = rloader.pose(loader.identity_in_raw(100))
    # print("Raw pose @ 0:", p0)
    # print("Raw pose @ 100:", p1)
    # pdiff = p1.homo().dot(np.linalg.inv(p0.homo()))
    # rdiff = Rotation.from_matrix(pdiff[:3,:3])
    # print("Rotation:", rdiff.as_euler("xyz"))

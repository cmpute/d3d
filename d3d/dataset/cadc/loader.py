import json
from itertools import chain
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import yaml
from addict import Dict as edict
from d3d.abstraction import TransformSet
from d3d.dataset.base import (TrackingDatasetBase, expand_idx, expand_idx_name,
                              split_trainval_seq)
from d3d.dataset.cadc import utils
from d3d.dataset.zip import PatchedZipFile
from sortedcontainers import SortedDict


class CADCDLoader(TrackingDatasetBase):
    """
    Load and parse CADC Dataset into a usable format, please organize the files into following structure

    * Zip Files::

        - <base_path_directory>
            - 2018_03_06
                - calib.zip
                - 0001
                    - labeled.zip
                    - raw.zip
                    - 3d_ann.json
                - 0002
                - ...
            - ...

    * Unzipped Structure::

        - <base_path directory>
            - 2018_03_06
                - calib
                    - ...
                - 0001
                    - labeled
                        - image_00
                        - image_01
                        - ...
                    - raw
                        - image_00
                        - image_01
                        - ...
                    - 3d_ann.json
                - 0002
                - ...

    For description of constructor parameters, please refer to :class:`d3d.dataset.base.TrackingDatasetBase`

    :param datatype: 'labeled' / 'raw'
    :type datatype: str
    """

    VALID_CAM_NAMES = ["camera_F", "camera_FR", "camera_RF", "camera_RB", "camera_B", "camera_LB", "camera_LF", "camera_FL"]
    VALID_LIDAR_NAMES = ["lidar"]
    VALID_OBJ_CLASSES = None
    _frame2folder = {
        "camera_F": "image_00", "camera_FR": "image_01", "camera_RF": "image_02", "camera_RB": "image_03",
        "camera_B": "image_04", "camera_LB": "image_05", "camera_LF": "image_06", "camera_FL": "image_07",
        "lidar": "lidar_points", "novatel": "novatel"
    }

    def __init__(self, base_path, datatype: str = 'labeled', inzip=True, phase="training",
                 trainval_split=1, trainval_random=False, trainval_byseq=False, nframes=0):
        super().__init__(base_path, inzip=inzip, phase=phase, nframes=nframes,
                         trainval_split=trainval_split, trainval_random=trainval_random,
                         trainval_byseq=trainval_byseq)
        self.datatype = datatype

        if phase == "testing":
            raise ValueError("There's no testing split for CADC dataset!")
        if datatype != "labeled":
            raise NotImplementedError("Currently only labeled data are supported!")

        # count total number of frames
        frame_count = dict()
        _dates = ["2018_03_06", "2018_03_07", "2019_02_27"]
        if self.inzip:
            globs = [self.base_path.glob(f"{date}/00*/{datatype}.zip")
                        for date in _dates]
            for archive in chain(*globs):
                with ZipFile(archive) as data:
                    velo_files = (name for name in data.namelist() if name.endswith(".bin"))

                    seq = '-'.join(archive.parent.parts[-2:])
                    frame_count[seq] = sum(1 for _ in velo_files)
        else:
            for date in _dates:
                if not (self.base_path / date).exists():
                    continue

                for archive in (self.base_path / date).iterdir():
                    if not archive.is_dir(): # skip calibration files
                        continue

                    seq = archive.name
                    frame_count[seq] = sum(1 for _ in (archive / "velodyne_points" / "data").iterdir())

        if not len(frame_count):
            raise ValueError("Cannot parse dataset or empty dataset, please check path, inzip option and file structure")
        self.frame_dict = SortedDict(frame_count)

        self.frames = split_trainval_seq(phase, self.frame_dict, trainval_split, trainval_random, trainval_byseq)
        self._label_cache = {} # used to store parsed label data
        self._calib_cache = {} # used to store parsed calibration data
        self._timestamp_cache = {} # used to store parsed timestamp
        self._pose_cache = {} # used to store parsed pose data
        self._3dann_cache = {} # used to store parsed tracklet data
        self._tracklet_mapping = {} # inverse mapping for tracklets

    def __len__(self):
        return len(self.frames)

    @property
    def sequence_ids(self):
        return list(self.frame_dict.keys())

    @property
    def sequence_sizes(self):
        return dict(self.frame_dict)

    def _split_seqid(self, seq_id):
        return seq_id[:10], seq_id[11:]

    def _locate_frame(self, idx):
        # use underlying frame index
        idx = self.frames[idx]

        for k, v in self.frame_dict.items():
            if idx < (v - self.nframes):
                return k, idx
            idx -= (v - self.nframes)
        raise ValueError("Index larger than dataset size")

    def _preload_calib(self, seq_id):
        date = self._split_seqid(seq_id)[0]
        if date in self._calib_cache:
            return

        calib = TransformSet("base_link")

        def add_cam_intrisic(data):
            data = edict(data)
            P = np.array(data.camera_matrix.data).reshape(3, 3)
            distorts = data.distortion_coefficients.data
            if len(distorts) == 4:
                distorts.append(0.0)
            calib.set_intrinsic_camera(data.camera_name, P, (data.image_width, data.image_height),
                distort_coeffs=data.distortion_coefficients.data, intri_matrix=P, rotate=False)

        def add_extrinsics(data):
            data = edict(data)
            calib.set_extrinsic(np.reshape(data.T_BASELINK_LIDAR, (4,4)), "base_link", "lidar")
            for i in range(8):
                calib.set_extrinsic(np.reshape(data['T_LIDAR_CAM%02d' % i], (4,4)), "lidar", self.VALID_CAM_NAMES[i])
            calib.set_extrinsic(np.reshape(data.T_00CAMERA_00IMU, (4,4)), 'camera_F', 'xsens_300')
            calib.set_extrinsic(np.reshape(data.T_03CAMERA_03IMU, (4,4)), 'camera_RB', 'xsens_30')
            calib.set_extrinsic(np.reshape(data.T_LIDAR_GPSIMU, (4,4)), 'lidar', 'novatel')

        # sensors with no intrinsic params
        calib.set_intrinsic_lidar("lidar")
        calib.set_intrinsic_general("novatel")
        calib.set_intrinsic_general("xsens_30")
        calib.set_intrinsic_general("xsens_300")

        if self.inzip:
            with ZipFile(self.base_path / date / "calib.zip") as source:
                for i in range(8):
                    add_cam_intrisic(yaml.safe_load(source.read("calib/%02d.yaml" % i)))
                add_extrinsics(yaml.safe_load(source.read("calib/extrinsics.yaml")))
        else:
            source = self.base_path / date
            for i in range(8):
                add_cam_intrisic(yaml.safe_load(source.read_bytes("calib/%02d.yaml" % i)))
            add_extrinsics(yaml.safe_load(source.read_bytes("calib/extrinsics.yaml")))

        self._calib_cache[date] = calib

    def calibration_data(self, idx, raw=False):
        assert not self._return_file_path, "The calibration is not stored in single file!"
        if isinstance(idx, int):
            seq_id, _ = self._locate_frame(idx)
        else:
            seq_id, _ = idx

        self._preload_calib(seq_id)
        date = self._split_seqid(seq_id)[0]
        return self._calib_cache[date]

    def _preload_timestamp(self, seq_id, name):
        date, drive = self._split_seqid(seq_id)
        if seq_id in self._timestamp_cache:
            return

        drive_path = self.base_path / date / drive
        tsdict = {}

        for frame, folder in self._frame2folder.items():
            fname = Path(self.datatype, folder, "timestamps.txt")
            if self.inzip:
                with PatchedZipFile(drive_path / f"{self.datatype}.zip", to_extract=fname) as data:
                    tsdict[frame] = utils.load_timestamps(data, fname).astype(int) // 1000
            else:
                tsdict[frame] = utils.load_timestamps(self.base_path, fname).astype(int) // 1000
        self._timestamp_cache[seq_id] = tsdict

    @expand_idx_name(VALID_CAM_NAMES + VALID_LIDAR_NAMES + ['novatel', 'xsens_30', 'xsens_300'])
    def timestamp(self, idx, names="lidar"):
        assert not self._return_file_path, "The timestamp is not stored in single file!"
        seq_id, frame_idx = idx
        self._preload_timestamp(seq_id, names)
        return self._timestamp_cache[seq_id][names][frame_idx]

    def _preload_ann_3d(self, seq_id):
        if seq_id in self._3dann_cache:
            return

        date, drive = self._split_seqid(seq_id)
        anno_file = self.base_path / date / drive / "3d_ann.json"
        with open(anno_file) as fin:
            data = json.load(fin)
        self._3dann_cache[seq_id] = data

    @expand_idx
    def annotation_3dobject(self, idx):
        assert not self._return_file_path, "The annotation is not stored in single file!"
        seq_id, frame_idx = idx
        self._preload_ann_3d(seq_id)
        return utils.load_3d_ann(self._3dann_cache[seq_id][frame_idx])

    @expand_idx
    def pose(self, idx, raw=False):
        seq_id, frame_idx = idx
        date, drive = self._split_seqid(seq_id)
        drive_path = self.base_path / date / drive

        file_name = Path(self.datatype, "novatel", "data", "%010d.txt" % frame_idx)
        drive_path = self.base_path / date / drive
        if self._return_file_path:
            return drive_path / file_name

        if self.inzip:
            with PatchedZipFile(drive_path / f"{self.datatype}.zip", to_extract=file_name) as source:
                data = utils.load_inspvax(source, file_name)
        else:
            data = utils.load_inspvax(drive_path, file_name)

        if raw:
            return data
        return utils.parse_pose_from_inspvax(data)

    @property
    def pose_name(self):
        return 'novatel'

    @expand_idx_name(VALID_CAM_NAMES)
    def camera_data(self, idx, names='camera_F'):
        seq_id, frame_idx = idx
        date, drive = self._split_seqid(seq_id)

        fname = Path(self.datatype, self._frame2folder[names], "data", "%010d.png" % frame_idx)
        drive_path = self.base_path / date / drive
        if self._return_file_path:
            return drive_path / fname

        if self.inzip:
            with PatchedZipFile(drive_path / f"{self.datatype}.zip", to_extract=fname) as source:
                return utils.load_image(source, fname)
        else:
            return utils.load_image(drive_path, fname)

    @expand_idx_name(VALID_LIDAR_NAMES)
    def lidar_data(self, idx, names='lidar', formatted=False):
        seq_id, frame_idx = idx
        date, drive = self._split_seqid(seq_id)

        fname = Path(self.datatype, "lidar_points", "data", "%010d.bin" % frame_idx)
        drive_path = self.base_path / date / drive
        if self._return_file_path:
            return drive_path / fname

        if self.inzip:
            with PatchedZipFile(drive_path / f"{self.datatype}.zip", to_extract=fname) as source:
                return utils.load_velo_scan(source, fname, formatted=formatted)
        else:
            return utils.load_velo_scan(drive_path, fname, formatted=formatted)

    @expand_idx
    def identity(self, idx):
        return idx

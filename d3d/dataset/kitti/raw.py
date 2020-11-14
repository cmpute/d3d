import numpy as np
from zipfile import ZipFile
from pathlib import Path
from itertools import chain
from collections import defaultdict, OrderedDict
from scipy.spatial.transform import Rotation

from d3d.abstraction import (ObjectTag, ObjectTarget3D, Target3DArray,
                             TransformSet)
from d3d.dataset.base import TrackingDatasetBase, check_frames, split_trainval, expand_idx, expand_idx_name
from d3d.dataset.kitti import utils
from d3d.dataset.kitti.utils import KittiObjectClass, OxtData
from d3d.dataset.zip import PatchedZipFile

class KittiRawDataset(TrackingDatasetBase):
    """
    Load and parse raw data into a usable format, please organize the files into following structure

    # Zip Files
    - 2011_09_26_calib.zip [required]
    - 2011_09_26_drive_0001_extract.zip
    - ...
    - 2011_09_26_drive_0001_sync.zip
    - ...
    - 2011_09_26_drive_0001_tracklets.zip
    - ...

    # Unzipped Structure
    - <base_path directory>
        - 2011_09_26
            - calib_cam_to_cam.txt
            - calib_imu_to_velo.txt
            - calib_velo_to_cam.txt
            - 2011_09_26_drive_0001_extract
                - image_00
                - image_01
                - image_02
                - image_03
                - oxts
                - velodyne_points
            - ...
            - 2011_09_26_drive_0001_sync
                - image_00
                - image_01
                - image_02
                - image_03
                - oxts
                - velodyne_points
                - tracklet_labels.xml
            - ...
            - 
    """

    VALID_CAM_NAMES = ["cam0", "cam1", "cam2", "cam3"]
    VALID_LIDAR_NAMES = ["velo"]
    FRAME2FOLDER = {
        "cam0": "image_00", "cam1": "image_01", "cam2": "image_02", "cam3": "image_03",
        "velo": "velodyne_points", "imu": "oxts"
    }

    def __init__(self, base_path, datatype='sync',
        inzip=True, phase="training",
        trainval_split=1, trainval_random=False,
        nframes=0
    ):
        """
        Set the path and pre-load calibration data and timestamps.

        # Parameters
        :param datatype: 'sync' (synced) / 'extract' (unsynced)
        """
        super().__init__(base_path, inzip=inzip, phase=phase, nframes=nframes,
                         trainval_split=trainval_split, trainval_random=trainval_random)
        self.datatype = datatype

        if phase == "testing":
            raise ValueError("There's no testing split for raw data!")
        if datatype != "sync":
            raise NotImplementedError("Currently only synced raw data are supported!")

        # count total number of frames
        frame_count = dict()
        _dates = ["2011_09_26", "2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03"]
        if self.inzip:
            globs = [self.base_path.glob(f"{date}_drive_*_{datatype}.zip")
                        for date in _dates]
            for archive in chain(*globs):
                with ZipFile(archive) as data:
                    velo_files = (name for name in data.namelist() if name.endswith(".bin"))

                    seq = archive.stem
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
        self.frame_dict = OrderedDict(frame_count)

        total_count = sum(frame_count.values()) - nframes * len(frame_count)
        self.frames = split_trainval(phase, total_count, trainval_split, trainval_random)
        self._label_cache = {} # used to store parsed label data
        self._calib_cache = {} # used to store parsed calibration data
        self._timestamp_cache = {} # used to store parsed timestamp
        self._pose_cache = {} # used to store parsed pose data
        self._tracklet_cache = {} # used to store parsed tracklet data
        self._tracklet_mapping = {} # inverse mapping for tracklets

    def __len__(self):
        return len(self.frames)

    @property
    def sequence_ids(self):
        return list(self.frame_dict.keys())

    @property
    def sequence_sizes(self):
        return dict(self.frame_dict)

    def _get_date(self, seq_id):
        return seq_id[:10]

    def _locate_frame(self, idx):
        # use underlying frame index
        idx = self.frames[idx]

        for k, v in self.frame_dict.items():
            if idx < (v - self.nframes):
                return k, idx
            idx -= (v - self.nframes)
        raise ValueError("Index larger than dataset size")

    def _preload_calib(self, seq_id):
        date = self._get_date(seq_id)
        if date in self._calib_cache:
            return

        if self.inzip:
            with ZipFile(self.base_path / f"{date}_calib.zip") as source:    
                self._calib_cache[date] = {
                    "cam_to_cam": utils.load_calib_file(source, f"{date}/calib_cam_to_cam.txt"),
                    "imu_to_velo": utils.load_calib_file(source, f"{date}/calib_imu_to_velo.txt"),
                    "velo_to_cam": utils.load_calib_file(source, f"{date}/calib_velo_to_cam.txt")
                }
        else:
            source = self.base_path / date
            self._calib_cache[date] = {
                    "cam_to_cam": utils.load_calib_file(source, "calib_cam_to_cam.txt"),
                    "imu_to_velo": utils.load_calib_file(source, "calib_imu_to_velo.txt"),
                    "velo_to_cam": utils.load_calib_file(source, "calib_velo_to_cam.txt")
                }

    def _load_calib(self, seq, raw=False):
        # load the calibration file data
        self._preload_calib(seq)
        date = self._get_date(seq)
        filedata = self._calib_cache[date]
        if raw:
            return filedata

        # load matrics
        data = TransformSet("velo")
        velo_to_cam = np.empty((3, 4))
        velo_to_cam[:3, :3] = filedata['velo_to_cam']['R'].reshape(3, 3)
        velo_to_cam[:3, 3] = filedata['velo_to_cam']['T']
        for i in range(4):
            S = filedata['cam_to_cam']['S_rect_%02d' % i].tolist()
            # TODO: here we have different R_rect's, what's the difference of them against the one used in object detection
            R = filedata['cam_to_cam']['R_rect_%02d' % i].reshape(3, 3)
            P = filedata['cam_to_cam']['P_rect_%02d' % i].reshape(3, 4)
            intri, offset = P[:, :3], P[:, 3]
            projection = intri.dot(R)
            offset_cartesian = np.linalg.inv(projection).dot(offset)
            extri = np.vstack([velo_to_cam, np.array([0,0,0,1])])
            extri[:3, 3] += offset_cartesian

            frame = "cam%d" % i
            data.set_intrinsic_camera(frame, projection, S, rotate=False)
            data.set_extrinsic(extri, frame_to=frame)

        imu_to_velo = np.empty((3, 4))
        imu_to_velo[:3, :3] = filedata['imu_to_velo']['R'].reshape(3, 3)
        imu_to_velo[:3, 3] = filedata['imu_to_velo']['T']
        data.set_intrinsic_general("imu")
        data.set_extrinsic(imu_to_velo, frame_from="imu")

        return data

    def calibration_data(self, idx, raw=False):
        if isinstance(idx, int):
            seq_id, _ = self._locate_frame(idx)
        else:
            seq_id, _ = idx

        return self._load_calib(seq_id, raw=raw)

    def _preload_timestamp(self, seq_id):
        date = self._get_date(seq_id)
        if seq_id in self._timestamp_cache:
            return

        tsdict = {}
        for frame, folder in self.FRAME2FOLDER.items():
            fname = Path(date, seq_id, folder, "timestamps.txt")
            if self.inzip:
                with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as data:
                    tsdict[frame] = utils.load_timestamps(data, fname, formatted=True)
            else:
                tsdict[frame] = utils.load_timestamps(self.base_path, fname, formatted=True)
        self._timestamp_cache[seq_id] = tsdict

    @expand_idx_name(VALID_CAM_NAMES + VALID_LIDAR_NAMES)
    def timestamp(self, idx, names="velo"):
        '''
        Get the timestamp of the data.

        :param names: specify the data source of timestamp. Options include {cam0-3, velo, imu}
        '''
        seq_id, frame_idx = idx
        self._preload_timestamp(seq_id)
        return self._timestamp_cache[seq_id][names][frame_idx].astype(int) // 1000

    def _preload_tracklets(self, seq_id):
        if seq_id in self._tracklet_cache:
            return

        date = self._get_date(seq_id)
        fname = Path(date, seq_id, "tracklet_labels.xml")
        if self.inzip:
            zname = seq_id[:-len(self.datatype)] + "tracklets"
            with ZipFile(self.base_path / f"{zname}.zip") as data:
                tracklets = utils.load_tracklets(data, fname)
        else:
            tracklets = utils.load_tracklets(self.base_path, fname)

        # inverse mapping
        objs = defaultdict(list) # (frame -> list of objects)
        for tid, tr in enumerate(tracklets):
            dim = [tr.l, tr.w, tr.h]
            tag = ObjectTag(tr.objectType, KittiObjectClass)
            for pose_idx, pose in enumerate(tr.poses):
                # TODO: convert from camera to lidar coordinate
                pos = [pose.tx, pose.ty, pose.tz]
                ori = Rotation.from_euler("ZYX", (pose.rz, pose.ry, pose.rx))
                objs[pose_idx + tr.first_frame].append(ObjectTarget3D(pos, ori, dim, tag, tid=tid))

        self._tracklet_cache[seq_id] = {k: Target3DArray(l, frame="velo") for k, l in objs.items()}

    @expand_idx
    def annotation_3dobject(self, idx):
        seq_id, frame_idx = idx
        self._preload_tracklets(seq_id)
        return self._tracklet_cache[seq_id][frame_idx]

    @expand_idx
    def pose(self, idx, raw=False):
        seq_id, frame_idx = idx
        date = self._get_date(seq_id)

        file_name = Path(date, seq_id, "oxts", "data", "%010d.txt" % frame_idx)
        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=file_name) as data:
                oxt = utils.load_oxt_file(data, file_name)[0]
        else:
            oxt = utils.load_oxt_file(self.base_path, file_name)[0]
        return utils.parse_pose_from_oxt(oxt)

    @expand_idx_name(VALID_CAM_NAMES)
    def camera_data(self, idx, names='cam2'):
        seq_id, frame_idx = idx
        date = self._get_date(seq_id)

        fname = Path(date, seq_id, self.FRAME2FOLDER[names], 'data', '%010d.png' % frame_idx)
        gray = names in ['cam0', 'cam1']
        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as source:
                return utils.load_image(source, fname, gray=gray)
        else:
            return utils.load_image(self.base_path, fname, gray=gray)

    @expand_idx_name(VALID_LIDAR_NAMES)
    def lidar_data(self, idx, names='velo'):
        seq_id, frame_idx = idx
        date = self._get_date(seq_id)

        fname = Path(date, seq_id, 'velodyne_points', 'data', '%010d.bin' % frame_idx)
        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip") as source:
                return utils.load_velo_scan(source, fname)
        else:
            return utils.load_velo_scan(self.base_path, fname)

    @expand_idx
    def identity(self, idx):
        return idx


from collections import OrderedDict
from itertools import chain
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from d3d.abstraction import (EgoPose, ObjectTag, ObjectTarget3D, Target3DArray,
                             TransformSet)
from d3d.dataset.base import (TrackingDatasetBase, check_frames, expand_idx,
                              expand_idx_name, split_trainval)
from d3d.dataset.kitti.utils import load_image, load_velo_scan, load_timestamps
from d3d.dataset.zip import PatchedZipFile
from d3d.dataset.kitti360.utils import Kitti360Class, load_bboxes, load_sick_scan, kittiId2label
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d


class KITTI360Loader(TrackingDatasetBase):
    """
    Load KITTI-360 dataset into a usable format.
    The dataset structure should follow the official documents.

    # Zip Files
    - calibration.zip
    - data_3d_bboxes.zip
    - data_3d_semantics.zip
    - data_poses.zip
    - data_timestamps_sick.zip
    - data_timestamps_velodyne.zip
    - 2013_05_28_drive_0000_sync_sick.zip
    - 2013_05_28_drive_0000_sync_velodyne.zip
    - ...

    # Unzipped Structure
    - <base_path directory>
        - calibration
        - data_2d_raw
            - 2013_05_28_drive_0000_sync
            - ...
        - data_2d_semantics
            - 2013_05_28_drive_0000_sync
            - ...
        - data_3d_raw
            - 2013_05_28_drive_0000_sync
            - ...
        - data_3d_semantics
            - 2013_05_28_drive_0000_sync
            - ...
    """
    VALID_CAM_NAMES = ['cam1', 'cam2', 'cam3', 'cam4'] # cam 1,2 are persective
    VALID_LIDAR_NAMES = ['velo', 'sick'] # velo stands for velodyne

    FRAME_PATH_MAP = dict(
        sick=("data_3d_raw", "sick_points"    , "data"     , "data_timestamps_perspective.zip"),
        velo=("data_3d_raw", "velodyne_points", "data"     , "data_timestamps_velodyne_points.zip"),
        cam1=("data_2d_raw", "image_00"       , "data_rect", "data_timestamps_perspective.zip"),
        cam2=("data_2d_raw", "image_01"       , "data_rect", "data_timestamps_perspective.zip"),
        cam3=("data_2d_raw", "image_02"       , "data_rgb" , "data_timestamps_perspective.zip"),
        cam4=("data_2d_raw", "image_03"       , "data_rgb" , "data_timestamps_perspective.zip"),
    )

    def __init__(self, base_path, phase="training", inzip=False,
                 trainval_split=1, trainval_random=False, nframes=0):
        """
        :param phase: training, validation or testing
        :param trainval_split: placeholder for interface compatibility with other loaders
        :param visible_range: range for visible objects. Objects beyond that distance will be removed when reporting
        """

        if phase not in ['training', 'validation', 'testing']:
            raise ValueError("Invalid phase tag")

        self.base_path = Path(base_path)
        self.inzip = inzip
        self.phase = phase
        self.nframes = nframes

        # count total number of frames
        frame_count = dict()
        _dates = ["2013_05_28"]
        if self.inzip:
            _archives = [
                ("sick", ".bin"), ("velodyne", ".bin"),
                ("image_00", ".png"), ("image_01", ".png"),
                ("image_02", ".png"), ("image_03", ".png")
            ]
            for aname, ext in _archives:
                globs = [self.base_path.glob(f"{date}_drive_*_sync_{aname}.zip")
                            for date in _dates]
                for archive in chain(*globs):
                    with ZipFile(archive) as data:
                        data_files = (name for name in data.namelist() if name.endswith(ext))

                        seq = archive.stem[:archive.stem.rfind("_")]
                        frame_count[seq] = sum(1 for _ in data_files)

                # successfully found
                if len(frame_count) > 0:
                    break
        else:
            _folders = [
                # ("data_3d_raw", "sick_points", "data"), # SICK points are out of synchronization
                ("data_3d_raw", "velodyne_points", "data"),
                ("data_2d_raw", "image_00", "data_rect"),
                ("data_2d_raw", "image_01", "data_rect"),
                ("data_2d_raw", "image_02", "data_rgb"),
                ("data_2d_raw", "image_03", "data_rgb"),
            ]
            for ftype, fname, dname in _folders:
                globs = [self.base_path.glob(f"{ftype}/{date}_drive_*_sync")
                            for date in _dates]
                for archive in chain(*globs):
                    if not archive.is_dir(): # skip calibration files
                        continue
                    if not (archive / fname / "data").exists():
                        continue

                    seq = archive.name
                    frame_count[seq] = sum(1 for _ in (archive / fname / dname).iterdir())

                # successfully found
                if len(frame_count) > 0:
                    break

        if not frame_count:
            raise ValueError("Cannot parse dataset, please check path, inzip option and file structure")
        self.frame_dict = OrderedDict(frame_count)

        total_count = sum(frame_count.values()) - nframes * len(frame_count)
        self.frames = split_trainval(phase, total_count, trainval_split, trainval_random)
        self._poses_idx = {} # store loaded poses indices
        self._poses_t = {} # pose translation
        self._poses_r = {} # pose rotation
        self._3dobjects_cache = {} # store loaded object labels
        self._3dobjects_mapping =  {} # store frame to objects mapping
        self._timestamp_cache = {} # store timestamps

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
            if idx < (v - self.nframes):
                return k, idx
            idx -= (v - self.nframes)
        raise ValueError("Index larger than dataset size")

    @expand_idx_name(VALID_CAM_NAMES)
    def camera_data(self, idx, names='cam1'):
        seq_id, frame_idx = idx

        _, folder_name, dname, _ = self.FRAME_PATH_MAP[names]
        fname = Path(seq_id, folder_name, dname, '%010d.png' % frame_idx)
        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}_{folder_name}.zip", to_extract=fname) as source:
                return load_image(source, fname, gray=False)
        else:
            return load_image(self.base_path / "data_2d_raw", fname, gray=False)

    @expand_idx_name(['velo'])
    def lidar_data(self, idx, names='velo'):
        seq_id, frame_idx = idx
        
        if names == "velo":
            # load velodyne points
            fname = Path(seq_id, "velodyne_points", "data", '%010d.bin' % frame_idx)
            if self.inzip:
                with PatchedZipFile(self.base_path / f"{seq_id}_velodyne.zip", to_extract=fname) as source:
                    return load_velo_scan(source, fname)
            else:
                return load_velo_scan(self.base_path / "data_3d_raw", fname)

        elif names == "sick":
            # TODO: sick data is not supported due to synchronization issue currently
            #       official way to accumulate SICK data is by linear interpolate between adjacent point clouds
            fname = Path(seq_id, "sick_points", "data", '%010d.bin' % frame_idx)
            if self.inzip:
                with PatchedZipFile(self.base_path / f"{seq_id}_sick.zip", to_extract=fname) as source:
                    return load_sick_scan(source, fname)
            else:
                return load_sick_scan(self.base_path / "data_3d_raw", fname)
            
    def _preload_3dobjects(self, seq_id):
        assert self.phase in ["training", "validation"], "Testing set doesn't contains label"
        if seq_id in self._3dobjects_mapping:
            return

        # load poses first to filter out far boxes
        self._preload_poses(seq_id)

        fname = Path("data_3d_bboxes", "train", f"{seq_id}.xml")
        if self.inzip:
            with PatchedZipFile(self.base_path / "data_3d_bboxes.zip", to_extract=fname) as source:
                objlist, fmap = load_bboxes(source, fname)
        else:
            objlist, fmap = load_bboxes(self.base_path, fname)
            
        self._3dobjects_cache[seq_id] = objlist
        self._3dobjects_mapping[seq_id] = fmap

    @expand_idx
    def annotation_3dobject(self, idx, raw=False):
        seq_id, frame_idx = idx
        self._preload_3dobjects(seq_id)
        objects = [self._3dobjects_cache[seq_id][i.data]
                    for i in self._3dobjects_mapping[seq_id][frame_idx]]
        if raw:
            return objects

        boxes = Target3DArray(frame="velo")
        # TODO: filter out boxes beyond range
        for box in objects:
            tag = ObjectTag(kittiId2label[box.semanticId].name, Kitti360Class)
            pos = box.transform[:3, 3]
            ori = Rotation.from_matrix(box.transform[:3, :3])
            tid = box.index
            dim = np.ones(3) # TODO: this is not provided by KITTI360 ??
            boxes.append(ObjectTarget3D(pos, ori, dim, tag, tid=tid))
        return boxes

    def calibration_data(self, idx):
        pass

    def annotation_3dsemantic(self, idx):
        pass

    def annotation_2dsemantic(self, idx):
        pass

    def _preload_timestamps(self, seq, name):
        if (seq, name) in self._timestamp_cache:
            return

        folder, subfolder, _, archive = self.FRAME_PATH_MAP[name]
        fname = Path(folder, seq, subfolder, "timestamps.txt")
        if self.inzip:
            with PatchedZipFile(self.base_path / archive, to_extract=fname) as data:
                ts = load_timestamps(data, fname, formatted=True)
        else:
            ts = load_timestamps(self.base_path, fname, formatted=True)

        self._timestamp_cache[(seq, name)] = ts.astype(int) // 1000

    @expand_idx
    def timestamp(self, idx, names="velo"):
        if names == "sick":
            raise NotImplementedError("Indexing for sick points are unavailable yet!")

        seq_id, frame_idx = idx
        self._preload_timestamps(seq_id, names)

        return self._timestamp_cache[(seq_id, names)][frame_idx]

    def _preload_poses(self, seq):
        if seq in self._poses_idx:
            return

        fname = Path("data_poses", seq, "poses.txt")
        if self.inzip:
            with PatchedZipFile(self.base_path / "data_poses.zip", to_extract=fname) as data:
                plist = np.loadtxt(data.open(str(fname)))
        else:
            plist = np.loadtxt(self.base_path / fname)

        # do interpolation
        pose_indices = plist[:, 0].astype(int)
        pose_matrices = plist[:, 1:].reshape(-1, 3, 4)
        positions = pose_matrices[:, :, 3]
        rotations = Rotation.from_matrix(pose_matrices[:, :, :3])
        
        ts_frame = "velo" # the frame used for timestamp extraction
        self._preload_timestamps(seq, ts_frame)
        timestamps = self._timestamp_cache[(seq, ts_frame)]

        fpos = interp1d(timestamps[pose_indices], positions, axis=0, fill_value="extrapolate")
        positions = fpos(timestamps)
        frot = interp1d(timestamps[pose_indices], rotations.as_rotvec(), axis=0, fill_value="extrapolate")
        rotations = frot(timestamps)
        
        self._poses_idx[seq] = set(pose_indices)
        self._poses_t[seq] = positions
        self._poses_r[seq] = Rotation.from_rotvec(rotations)

    @expand_idx
    def pose(self, idx, interpolate=True):
        """
        :param interpolate: Not all frames contain pose data in KITTI-360. The loader
            returns interpolated pose if this param is set as True, otherwise returns None
        """
        seq_id, frame_idx = idx

        self._preload_poses(seq_id)
        if frame_idx not in self._poses_idx[seq_id] and not interpolate:
            return None

        return EgoPose(self._poses_t[seq_id][frame_idx], self._poses_r[seq_id][frame_idx])

if __name__ == "__main__":
    # loader = KITTI360Loader("/mnt/storage8t/datasets/KITTI-360/", inzip=True)
    loader = KITTI360Loader("/media/jacob/Storage/Datasets/kitti360", inzip=False, nframes=0)
    # print(loader.camera_data(0))
    # print(loader.lidar_data(0))
    for i in range(50):
        print(loader.pose(i))
        print(loader.annotation_3dobject(i))
    # print(loader.timestamp(1))
    # for obj in loader.annotation_3dobject(174, raw=True):
    #     print(obj.vertices)
    # print(loader.annotation_3dobject(174))

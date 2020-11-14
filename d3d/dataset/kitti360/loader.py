from collections import OrderedDict
from itertools import chain
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from d3d.abstraction import (EgoPose, ObjectTag, ObjectTarget3D, Target3DArray,
                             TransformSet)
from d3d.dataset.base import (TrackingDatasetBase, check_frames, expand_idx,
                              expand_idx_name, split_trainval)
from d3d.dataset.kitti.utils import load_image, load_velo_scan
from d3d.dataset.zip import PatchedZipFile
from PIL import Image
from scipy.spatial.transform import Rotation


def load_sick_scan(basepath, file):
    if isinstance(basepath, (str, Path)):
        scan = np.fromfile(Path(basepath, file), dtype=np.float32)
    else:
        with basepath.open(str(file)) as fin:
            buffer = fin.read()
        scan = np.frombuffer(buffer, dtype=np.float32)
    return scan.reshape((-1, 2))

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

    def __init__(self, base_path, phase="training", inzip=False, trainval_split=1, trainval_random=False, nframes=0):
        """
        :param phase: training, validation or testing
        :param trainval_split: placeholder for interface compatibility with other loaders
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
        self._poses_idx = {} # store loaded poses array
        self._poses_matrix = {}

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

        _folders = {
            "cam1": ("image_00", "data_rect"),
            "cam2": ("image_01", "data_rect"),
            "cam3": ("image_02", "data_rgb"),
            "cam4": ("image_03", "data_rgb")
        }

        folder_name, dname = _folders[names]
        fname = Path(seq_id, folder_name, dname, '%010d.png' % frame_idx)
        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}_{folder_name}.zip", to_extract=fname) as source:
                return load_image(source, fname, gray=False)
        else:
            return load_image(self.base_path / "data_2d_raw", fname, gray=False)

    @expand_idx_name(['velo'])
    # TODO: sick data is not supported due to synchronization issue currently
    #       official way to accumulate SICK data is by linear interpolate between adjacent point clouds
    def lidar_data(self, idx, names='velo', concat=False): # TODO(v0.4): implement concatenation
        seq_id, frame_idx = idx
        
        if names == "velo":
            # load velodyne points
            fname = Path(seq_id, "velodyne_points", "data", '%010d.bin' % frame_idx)
            if self.inzip:
                with PatchedZipFile(self.base_path / f"{seq_id}_velodyne.zip", to_extract=fname) as source:
                    return load_velo_scan(source, fname)
            else:
                return load_velo_scan(self.base_path / "data_3d_raw", fname)

    def annotation_3dobject(self, idx):
        pass

    def calibration_data(self, idx):
        pass

    def annotation_3dsemantic(self, idx):
        pass

    def annotation_2dsemantic(self, idx):
        pass

    def timestamp(self, idx, name="velo"):
        pass

    def _preload_poses(self, seq):
        if seq in self._poses_idx:
            return

        fname = Path("data_poses", seq, "poses.txt")
        if self.inzip:
            with PatchedZipFile(self.base_path / "data_poses.zip", to_extract=fname) as data:
            # with ZipFile(self.base_path / "data_poses.zip") as data:
                plist = np.loadtxt(data.open(str(fname)))
        else:
            plist = np.loadtxt(self.base_path / fname)

        self._poses_idx[seq] = plist[:, 0].astype(int)
        self._poses_matrix[seq] = plist[:, 1:].reshape(-1, 3, 4)

    @expand_idx
    def pose(self, idx, interpolate=False):
        """
        :param interpolate: Not all frames contain pose data in KITTI-360. The loader
            returns interpolated pose if this param is set as True, otherwise returns None
        """
        seq_id, frame_idx = idx

        self._preload_poses(seq_id)
        pidx = np.searchsorted(self._poses_idx[seq_id], frame_idx)
        if frame_idx != self._poses_idx[seq_id][pidx]:
            if interpolate:
                raise NotImplementedError() # TODO: add interpolation ability
            else:
                return None

        rt = self._poses_matrix[seq_id][pidx]
        t = rt[:3, 3]
        r = Rotation.from_matrix(rt[:3, :3])
        return EgoPose(t, r)

if __name__ == "__main__":
    loader = KITTI360Loader("/mnt/storage8t/datasets/KITTI-360/", inzip=True)
    # loader = KITTI360Loader("/mnt/storage8t/jacobz/kitti360_extracted", inzip=False, nframes=0)
    print(loader.camera_data(0))
    print(loader.lidar_data(0))
    print(loader.pose(1))

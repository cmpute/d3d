import numpy as np
from zipfile import ZipFile
from pathlib import Path
from itertools import chain
from collections import OrderedDict
from d3d.abstraction import (ObjectTag, ObjectTarget3D, Target3DArray,
                             TransformSet)
from d3d.dataset.base import TrackingDatasetBase, check_frames, split_trainval
from d3d.dataset.kitti.utils import load_image, load_velo_scan
from PIL import Image

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

    def __len__(self):
        return len(self.frames)

    @property
    def sequence_ids(self):
        return list(self.frame_dict.keys())

    @property
    def sequence_sizes(self):
        return dict(self.frame_dict)

    def _locate_frame(self, idx): # TODO(v0.4): this function is duplicated for multiple times
        # use underlying frame index
        idx = self.frames[idx]

        for k, v in self.frame_dict.items():
            if (idx + self.nframes) < v:
                return k, (idx + self.nframes)
            idx -= (v - self.nframes)
        raise ValueError("Index larger than dataset size")

    def camera_data(self, idx, names='cam1'):
        if isinstance(idx, int):
            seq_id, frame_idx = self._locate_frame(idx)
        else:
            seq_id, frame_idx= idx
        unpack_result, names = check_frames(names, self.VALID_CAM_NAMES)
        
        outputs = []
        _folders = {
            "cam1": ("image_00", "data_rect"),
            "cam2": ("image_01", "data_rect"),
            "cam3": ("image_02", "data_rgb"),
            "cam4": ("image_03", "data_rgb")
        }
        for name in names:
            folder_name, dname = _folders[name]
            if self.inzip:
                source = ZipFile(self.base_path / f"{seq_id}_image_{folder_name}.zip")

            data_seq = []
            for fidx in range(frame_idx - self.nframes, frame_idx+1):
                file_name = Path(seq_id, folder_name, dname, '%010d.png' % fidx)
                if self.inzip:
                    image = utils.load_image(source, file_name, gray=False)
                else:
                    image = load_image(self.base_path / "data_2d_raw", file_name, gray=False)
                data_seq.append(image)
            if self.nframes == 0: # unpack if only 1 frame
                data_seq = data_seq[0]

            outputs.append(data_seq)
            if self.inzip:
                source.close()

        return outputs[0] if unpack_result else outputs

    def lidar_data(self, idx, names='velo', concat=False): # TODO(v4.0): implement concatenation
        if isinstance(idx, int): # this snippet has also been duplicate for many times
            seq_id, frame_idx = self._locate_frame(idx)
        else:
            seq_id, frame_idx = idx
        # TODO: sick data is not supported due to synchronization issue currently
        #       official way to accumulate SICK data is by linear interpolate between adjacent point clouds
        # unpack_result, names = check_frames(names, self.VALID_LIDAR_NAMES)
        unpack_result, names = check_frames(names, ["velo"])

        # load velodyne points
        data_seq = []
        if self.inzip:
            source = ZipFile(self.base_path / f"{seq_id}_velodyne.zip")
        for fidx in range(frame_idx - self.nframes, frame_idx+1):
            fname = Path(seq_id, "velodyne_points", "data", '%010d.bin' % fidx)
            if self.inzip:
                data_seq.append(load_velo_scan(source, fname))
            else:
                data_seq.append(load_velo_scan(self.base_path / "data_3d_raw", fname))
        if self.nframes == 0: # unpack if only 1 frame
            data_seq = data_seq[0]

        if self.inzip:
            source.close()
        return data_seq

    def lidar_objects(self, idx):
        pass

    def calibration_data(self, idx):
        pass

    def annotation_3dsemantic(self, idx):
        pass

    def annotation_2dsemantic(self, idx):
        pass

    def timestamp(self, idx, frame="velo"):
        pass

    def pose(self, idx, interpolate=False):
        """
        :param interpolate: Not all frames contain pose data in KITTI-360. The loader
            returns interpolated pose if this param is set as True, otherwise returns None
        """
        pass

if __name__ == "__main__":
    # loader = KITTI360Loader("/mnt/storage8t/datasets/KITTI-360/", inzip=True)
    loader = KITTI360Loader("/mnt/storage8t/jacobz/kitti360_extracted", inzip=False, nframes=0)
    print(loader.camera_data(0))
    print(loader.lidar_data(0))

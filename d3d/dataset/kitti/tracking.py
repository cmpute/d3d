import numpy as np
from zipfile import ZipFile
from pathlib import Path
from collections import defaultdict
from scipy.spatial.transform import Rotation

from d3d.abstraction import (ObjectTag, ObjectTarget3D, ObjectTarget3DArray,
                             TransformSet)
from d3d.dataset.base import DetectionDatasetBase, check_frames, split_trainval
from d3d.dataset.kitti import utils

class KittiObjectLoader(DetectionDatasetBase):
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

    def __init__(self, base_path, inzip=False, phase="training", trainval_split=0.8, trainval_random=False):
        base_path = Path(base_path)
        self.base_path = base_path
        self.inzip = inzip
        self.phase = phase
        self.phase_path = 'training' if phase == 'validation' else phase

        if phase not in ['training', 'validation', 'testing']:
            raise ValueError("Invalid phase tag")

        # count total number of frames
        seq_count = None
        frame_count = defaultdict(int)
        if self.inzip:
            for folder in ['label_2', 'image_2', 'image_3', 'velodyne']:
                data_zip = base_path / ("data_tracking_%s.zip" % folder)
                if data_zip.exists():
                    seq_count = 0
                    with ZipFile(data_zip) as data:
                        for name in data.namelist():
                            phase, _, seq, frame = Path(name).parts
                            if phase != self.phase_path:
                                continue

                            seq = int(seq)
                            seq_count = max(seq, seq_count)
                            frame_count[seq] = max(frame_count[seq], int(frame))
                    break
        else:
            for folder in ['label_02', 'image_02', 'image_03', 'velodyne']:
                fpath = base_path / self.phase_path / folder
                if fpath.exists():
                    seq_count = 0
                    for seq_path in fpath.iterdir():
                        frame_count[seq_count] = sum(1 for _ in seq_path.iterdir())
                        seq_count += 1
                    break

        self.frames = split_trainval(phase, total_count, trainval_split, trainval_random)
        self._image_size_cache = {} # used to store the image size

    def __len__(self):
        return len(self.frames)

    # def camera_data(self, idx, names='cam2'):
    #     unpack_result, names = check_frames(names, self.VALID_CAM_NAMES)
        
    #     outputs = []
    #     for name in names:
    #         if name == "cam2":
    #             folder_name = "image_2"
    #         elif name == "cam3":
    #             folder_name = "image_3"

    #         file_name = self.phase_path / folder_name / ('%06d.png' % self.frames[idx])
    #         if self.inzip:
    #             with ZipFile(self.base_path / ("data_object_%s.zip" % folder_name)) as source:
    #                 image = utils.load_image(source, file_name, gray=False)
    #         else:
    #             image = utils.load_image(self.base_path, file_name, gray=False)

    #         outputs.append(image)
    #         if idx not in self._image_size_cache:
    #             self._image_size_cache[idx] = image.size

    #     if unpack_result:
    #         return outputs[0]
    #     else:
    #         return outputs

    # def lidar_data(self, idx, names='velo', concat=True):
    #     if isinstance(names, str):
    #         names = [names]
    #     if names != self.VALID_LIDAR_NAMES:
    #         raise ValueError("There's only one lidar in KITTI dataset")

    #     fname = self.phase_path / 'velodyne' / ('%06d.bin' % self.frames[idx])
    #     if self.inzip:
    #         with ZipFile(self.base_path / "data_object_velodyne.zip") as source:
    #             return utils.load_velo_scan(source, fname)
    #     else:
    #         return utils.load_velo_scan(self.base_path, fname)
    #     source.close()

    # def calibration_data(self, idx, raw=False):
    #     if self.inzip:
    #         with ZipFile(self.base_path / "data_object_calib.zip") as source:
    #             return self._load_calib(source, idx, raw)
    #     else:
    #         return self._load_calib(self.base_path, idx, raw)

    # def lidar_objects(self, idx, raw=False):
    #     '''
    #     Return list of converted ground truth targets in lidar frame.

    #     Note that Objects labelled as `DontCare` are removed
    #     '''

    #     if self.phase_path.name == "testing":
    #         raise ValueError("Testing dataset doesn't contain label data")

    #     fname = self.phase_path / 'label_2' / ('%06d.txt' % self.frames[idx])
    #     if self.inzip:
    #         with ZipFile(self.base_path / "data_object_label_2.zip") as source:
    #             label = utils.load_label(source, fname)
    #     else:
    #         label = utils.load_label(self.base_path, fname)

    #     if raw:
    #         return label
    #     return parse_label(label, self.calibration_data(idx, raw=True))

    # def identity(self, idx):
    #     '''
    #     For KITTI this method just return the phase and index
    #     '''
    #     return self.phase_path.name, self.frames[idx]

    # def _load_calib(self, basepath, idx, raw=False):
    #     data = TransformSet("velo")

    #     # load the calibration file
    #     filename = self.phase_path / 'calib' / ('%06d.txt' % self.frames[idx])
    #     filedata = utils.load_calib_file(basepath, filename)

    #     if raw:
    #         return filedata

    #     # load image size, which could take additional time
    #     if idx not in self._image_size_cache:
    #         self.camera_data(idx) # fill image size cache
    #     image_size = self._image_size_cache[idx]

    #     # load matrics
    #     rect = filedata['R0_rect'].reshape(3, 3)
    #     velo_to_cam = filedata['Tr_velo_to_cam'].reshape(3, 4)
    #     for i in range(4):
    #         P = filedata['P%d' % i].reshape(3, 4)
    #         intri, offset = P[:, :3], P[:, 3]
    #         projection = intri.dot(rect)
    #         offset_cartesian = np.linalg.inv(projection).dot(offset)
    #         extri = np.vstack([velo_to_cam, np.array([0,0,0,1])])
    #         extri[:3, 3] += offset_cartesian

    #         frame = "cam%d" % i
    #         data.set_intrinsic_camera(frame, projection, image_size, rotate=False)
    #         data.set_extrinsic(extri, frame_to=frame)

    #     data.set_intrinsic_general("imu")
    #     data.set_extrinsic(filedata['Tr_imu_to_velo'].reshape(3, 4), frame_from="imu")

    #     return data
import base64
import itertools
import json
import logging
import os
import os.path as osp
import shutil
import struct
import subprocess
import tempfile
import zipfile
import tarfile
from pathlib import Path
from collections import OrderedDict
from enum import Enum, auto
from io import BytesIO

import numpy as np
import msgpack
from addict import Dict as edict
from PIL import Image
from scipy.spatial.transform import Rotation

from d3d.abstraction import (ObjectTag, ObjectTarget3D, Target3DArray,
                             TransformSet, EgoPose)
from d3d.dataset.zip import PatchedZipFile
from d3d.dataset.base import TrackingDatasetBase, check_frames, expand_idx, expand_idx_name

_logger = logging.getLogger("d3d")

class WaymoObjectClass(Enum):
    '''
    Category of objects in KITTI dataset
    '''
    Unknown = 0
    Vehicle = auto()
    Pedestrian = auto()
    Sign = auto()
    Cyclist = auto()

class WaymoLoader(TrackingDatasetBase):
    """
    Load Waymo dataset into a usable format.
    Please use the d3d_waymo_convert command to convert the dataset first into following formats

    * Directory Structure::

        - <base_path directory>
            - training
                - xxxxxxxxxxxxxxxxxxxx_xxx_xxx_xxx_xxx(.zip)
                - ...
            - validation
                - xxxxxxxxxxxxxxxxxxxx_xxx_xxx_xxx_xxx(.zip)
                - ...

    For description of constructor parameters, please refer to :class:`d3d.dataset.base.TrackingDatasetBase`
    """
    VALID_CAM_NAMES = ["camera_front", "camera_front_left", "camera_front_right", "camera_side_left", "camera_side_right"]
    VALID_LIDAR_NAMES = ["lidar_top", "lidar_front", "lidar_side_left", "lidar_side_right", "lidar_rear"]
    VALID_OBJ_CLASSES = WaymoObjectClass

    def __init__(self, base_path, phase="training", inzip=False, trainval_split=None, trainval_random=False, nframes=0):
        super().__init__(base_path, inzip=inzip, phase=phase, nframes=nframes,
                         trainval_split=trainval_split, trainval_random=trainval_random)

        self.base_path = Path(base_path) / phase
        self.inzip = inzip
        self._load_metadata()

    def _load_metadata(self):
        meta_path = self.base_path / "metadata.msg"
        if not meta_path.exists():
            _logger.info("Creating metadata of Waymo dataset (%s)...", self.phase)
            metadata = {}

            if self.inzip:
                for archive in self.base_path.iterdir():
                    if archive.is_dir() or archive.suffix != ".zip":
                        continue

                    with PatchedZipFile(archive, to_extract="context/stats.json") as ar:
                        metadata[archive.stem] = json.loads(ar.read("context/stats.json"))
            else:
                for folder in self.base_path.iterdir():
                    if not folder.is_dir():
                        continue

                    with (folder / "context/stats.json").open() as fin:
                        metadata[folder.name] = json.load(fin)

            with open(meta_path, "wb") as fout:
                msgpack.pack(metadata, fout)

        with open(meta_path, "rb") as fin:
            self._metadata = OrderedDict()
            meta_json = msgpack.unpack(fin)
            for k, v in meta_json.items():
                self._metadata[k] = edict(v)

    def __len__(self):
        return sum(v.frame_count for v in self._metadata.values())

    def _locate_frame(self, idx):
        for k, v in self._metadata.items():
            if idx < v.frame_count:
                return k, idx
            idx -= v.frame_count
        raise ValueError("Index larger than dataset size")

    @expand_idx_name(VALID_LIDAR_NAMES)
    def lidar_data(self, idx, names=None):
        # XXX: support return ri2 data
        seq_id, frame_idx = idx

        fname = "%s/%04d.bin" % (names, frame_idx)
        if self._return_file_path:
            return self.base_path / seq_id / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / (seq_id + ".zip"), to_extract=fname) as ar:
                cloud = np.frombuffer(ar.read(fname), dtype='f4')
        else:
            cloud = np.fromfile(self.base_path / seq_id / fname, dtype='f4')
        cloud = cloud.reshape(-1, 5) # x, y, z, intensity, elongation

        # point cloud is represented in base frame in Waymo dataset
        calib = self.calibration_data(idx)
        rt = calib.extrinsics[names]
        cloud[:,:3] = cloud[:,:3].dot(rt[:3,:3].T) + rt[:3, 3]
        return cloud

    @expand_idx_name(VALID_CAM_NAMES)
    def camera_data(self, idx, names=None):
        """
        :param names: frame names of camera to be loaded
        """
        seq_id, frame_idx = idx
        fname = "%s/%04d.jpg" % (names, frame_idx)
        if self._return_file_path:
            return self.base_path / seq_id / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / (seq_id + ".zip"), to_extract=fname) as ar:
                return Image.open(ar.open(fname)).convert('RGB')
        else:
            Image.open(self.base_path / seq_id / fname).convert('RGB')

    @expand_idx_name(VALID_CAM_NAMES)
    def annotation_2dobject(self, idx, names=None):
        seq_id, frame_idx = idx

        fname = "label_%s/%04d.json" % (names, frame_idx)
        if self._return_file_path:
            return self.base_path / seq_id / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / (seq_id + ".zip"), to_extract=fname) as ar:
                data = json.loads(ar.read(fname))
        else:
            with (self.base_path / seq_id / fname).open() as fin:
                data = json.load(fin)
        labels = list(map(edict, data))

        # XXX: currently we don't have a interface for storing 2d object
        return labels

    @expand_idx
    def annotation_3dobject(self, idx, raw=False):
        seq_id, frame_idx = idx

        fname = "label_lidars/%04d.json" % frame_idx
        if self._return_file_path:
            return self.base_path / seq_id / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / (seq_id + ".zip"), to_extract=fname) as ar:
                data = json.loads(ar.read(fname))
        else:
            with (self.base_path / seq_id / fname).open() as fin:
                data = json.load(fin)
        labels = list(map(edict, data))
        if raw:
            return labels

        arr = Target3DArray(frame="vehicle") # or frame=None
        for label in labels:
            tid = base64.urlsafe_b64decode(label.id[:12])
            tid, = struct.unpack('Q', tid[:8])
            target = ObjectTarget3D(
                label.center,
                Rotation.from_euler("z", label.heading),
                label.size,
                ObjectTag(label.label, WaymoObjectClass),
                tid=tid
            )
            arr.append(target)

        return arr

    def calibration_data(self, idx):
        # TODO: add motion compensation, ref: https://github.com/waymo-research/waymo-open-dataset/issues/146
        # https://github.com/waymo-research/waymo-open-dataset/issues/79
        if isinstance(idx, int):
            seq_id, _ = self._locate_frame(idx)
        else:
            seq_id, _ = idx
        assert not self._return_file_path, "The calibration data is not in a single file!"

        calib_params = TransformSet("vehicle")

        # load json files
        fname_cams = "context/calib_cams.json"
        fname_lidars = "context/calib_lidars.json"
        if self.inzip:
            with PatchedZipFile(self.base_path / (seq_id + ".zip"), to_extract=[fname_cams, fname_lidars]) as ar:
                calib_cams = json.loads(ar.read(fname_cams))
                calib_lidars = json.loads(ar.read(fname_lidars))
        else:
            with (self.base_path / seq_id / fname_cams).open() as fin:
                calib_cams = json.load(fin)
            with (self.base_path / seq_id / fname_lidars).open() as fin:
                calib_lidars = json.load(fin)

        # parse camera calibration
        for frame, calib in calib_cams.items():
            frame = "camera_" + frame
            (fu, fv, cu, cv), distort = calib['intrinsic'][:4], calib['intrinsic'][4:]
            transform = np.array(calib['extrinsic']).reshape(4,4)
            size = (calib['width'], calib['height'])
            calib_params.set_intrinsic_pinhole(frame, size, cu, cv, fu, fv, distort_coeffs=distort)
            calib_params.set_extrinsic(transform, frame_from=frame)

        # parse lidar calibration
        for frame, calib in calib_lidars.items():
            frame = "lidar_" + frame
            calib_params.set_intrinsic_lidar(frame)
            transform = np.array(calib['extrinsic']).reshape(4,4)
            calib_params.set_extrinsic(transform, frame_from=frame)

        return calib_params

    @expand_idx
    def identity(self, idx):
        return idx

    @expand_idx
    def timestamp(self, idx):
        seq_id, frame_idx = idx
        fname = "timestamp/%04d.txt" % frame_idx
        if self.inzip:
            with PatchedZipFile(self.base_path / (seq_id + ".zip"), to_extract=fname) as ar:
                return int(ar.read(fname).decode())
        else:
            return int((self.base_path / seq_id / fname).read_bytes())

    @expand_idx
    def pose(self, idx, raw=False):
        seq_id, frame_idx = idx
        fname = "pose/%04d.bin" % frame_idx
        if self.inzip:
            with PatchedZipFile(self.base_path / (seq_id + ".zip"), to_extract=fname) as ar:
                rt = np.frombuffer(ar.read(fname), dtype="f8")
        else:
            rt = np.fromfile(self.base_path / seq_id / fname, dtype="f8")

        if raw:
            return rt
        return EgoPose(-rt[:3, 3], Rotation.from_matrix(rt[:3, :3]))

    @property
    def sequence_ids(self):
        return list(self._metadata.keys())

    @property
    def sequence_sizes(self):
        return {k: v.frame_count for k, v in self._metadata.items()}

def dump_detection_output(detections: Target3DArray, context: str, timestamp: int):
    '''
    :param detections: detection result
    :param ids: auxiliary information for output, each item contains context name and timestamp
    '''
    try:
        from waymo_open_dataset import label_pb2
        from waymo_open_dataset.protos import metrics_pb2
    except:
        _logger.error("Cannot find waymo_open_dataset, install the package at "
            "https://github.com/waymo-research/waymo-open-dataset, output will be skipped now.")
        return

    label_map = {
        WaymoObjectClass.Unknown: label_pb2.Label.TYPE_UNKNOWN,
        WaymoObjectClass.Vehicle: label_pb2.Label.TYPE_VEHICLE,
        WaymoObjectClass.Pedestrian: label_pb2.Label.TYPE_PEDESTRIAN,
        WaymoObjectClass.Sign: label_pb2.Label.TYPE_SIGN,
        WaymoObjectClass.Cyclist: label_pb2.Label.TYPE_CYCLIST
    }

    waymo_array = metrics_pb2.Objects()
    for target in detections:
        waymo_target = metrics_pb2.Object()

        # convert box parameters
        box = label_pb2.Label.Box()
        box.center_x = target.position[0]
        box.center_y = target.position[1]
        box.center_z = target.position[2]
        box.length = target.dimension[0]
        box.width = target.dimension[1]
        box.height = target.dimension[2]
        box.heading = target.yaw
        waymo_target.object.box.CopyFrom(box)

        # convert label
        waymo_target.object.type = label_map[target.tag_top]
        waymo_target.score = target.tag.scores[0]

        waymo_target.context_name = context
        waymo_target.frame_timestamp_micros = int(timestamp * 1e6)
        waymo_array.objects.append(waymo_target)

    return waymo_array

def execute_official_evaluator(exec_path, label_path, result_path, output_path, model_name=None, show_output=True):
    '''
    Execute compute_detection_metrics_main from waymo_open_dataset
    :param exec_path: path to compute_detection_metrics_main
    '''
    raise NotImplementedError()

def create_submission(exec_path, result_path, output_path, meta_path, model_name=None):
    '''
    Execute create_submission from waymo_open_dataset
    :param exec_path: path to create_submission
    :param result_path: path (or list of path) to detection result in binary protobuf
    :param meta_path: path to the metadata file (example: waymo_open_dataset/metrics/tools/submission.txtpb)
    :param output_path: output path for the created submission archive
    '''
    temp_path = tempfile.mkdtemp() + '/'
    model_name = model_name or "noname"
    cwd_path = temp_path + 'input' # change input directory
    os.mkdir(cwd_path)

    # combine single results
    if isinstance(result_path, str):
        result_path = [result_path]
    else:
        assert isinstance(result_path, (list, tuple))
    input_files, counter = [], 0
    print("Combining outputs into %s..." % temp_path)
    for rpath in result_path:
        from waymo_open_dataset.protos.metrics_pb2 import Objects
        
        # merge objects
        combined_objects = Objects()
        for f in os.listdir(rpath):
            with open(osp.join(rpath, f), "rb") as fin:
                objects = Objects()
                objects.ParseFromString(fin.read())
                combined_objects.MergeFrom(objects)
            
            if len(combined_objects.objects) > 1024: # create binary file every 1024 objects
                with open(osp.join(cwd_path, "%x.bin" % counter), "wb") as fout:
                    fout.write(combined_objects.SerializeToString())
                combined_objects = Objects()
                counter += 1

    # write remaining objects
    if len(combined_objects.objects) > 0:
        with open(osp.join(cwd_path, "%x.bin" % counter), "wb") as fout:
            fout.write(combined_objects.SerializeToString())
    input_files = ','.join(os.listdir(cwd_path))

    # create submissions
    print("Creating submission...")
    proc = subprocess.Popen([exec_path,
        "--input_filenames=%s" % input_files,
        "--output_filename=%s" % (temp_path + model_name),
        "--submission_filename=%s" % meta_path
    ], cwd=cwd_path)
    proc.wait()

    # create tarball
    print("Clean up...")
    if cwd_path != result_path: # remove combined files before zipping
        shutil.rmtree(cwd_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with tarfile.open(osp.join(output_path, model_name + ".tgz"), "w:gz") as tar:
        tar.add(temp_path, arcname=os.path.basename(temp_path))

    # clean
    shutil.rmtree(temp_path)

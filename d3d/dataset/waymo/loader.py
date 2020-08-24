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
from enum import Enum
from io import BytesIO

import numpy as np
from addict import Dict as edict
from PIL import Image
from scipy.spatial.transform import Rotation

from d3d.abstraction import (ObjectTag, ObjectTarget3D, Target3DArray,
                             TransformSet)
from d3d.dataset.base import TrackingDatasetBase, check_frames

_logger = logging.getLogger("d3d")

class WaymoObjectClass(Enum):
    Unknown = 0
    Vehicle = 1
    Pedestrian = 2
    Sign = 3
    Cyclist = 4

class WaymoObjectLoader(TrackingDatasetBase):
    """
    Load waymo dataset into a usable format.
    Please use the d3d_waymo_convert command to convert the dataset first into following formats

    # Directory Structure
    - <base_path directory>
        - training
            - xxxxxxxxxxxxxxxxxxxx_xxx_xxx_xxx_xxx.zip
            - ...
        - validation
            - xxxxxxxxxxxxxxxxxxxx_xxx_xxx_xxx_xxx.zip
            - ...
    """
    VALID_CAM_NAMES = ["camera_front", "camera_front_left", "camera_front_right", "camera_side_left", "camera_side_right"]
    VALID_LIDAR_NAMES = ["lidar_top", "lidar_front", "lidar_side_left", "lidar_side_right", "lidar_rear"]

    def __init__(self, base_path, phase="training", inzip=False, trainval_split=None, trainval_random=False, nframes=0):
        """
        :param phase: training, validation or testing
        :param trainval_split: placeholder for interface compatibility with other loaders
        """

        if not inzip:
            raise NotImplementedError("Currently only support load from zip files in Waymo dataset")
        if phase not in ['training', 'validation', 'testing']:
            raise ValueError("Invalid phase tag")

        self.base_path = Path(base_path) / phase
        self.phase = phase
        self.nframes = nframes
        self._load_metadata()

    def _load_metadata(self):
        meta_path = self.base_path / "metadata.json"
        if not meta_path.exists():
            _logger.info("Creating metadata of Waymo dataset (%s)...", self.phase)
            metadata = {}

            for archive in self.base_path.iterdir():
                if archive.is_dir() or archive.suffix != ".zip":
                    continue

                with zipfile.ZipFile(archive) as ar:
                    with ar.open("context/stats.json") as fin:
                        metadata[archive.stem] = json.loads(fin.read().decode())
            with open(meta_path, "w") as fout:
                json.dump(metadata, fout)

        with open(meta_path) as fin:
            self._metadata = OrderedDict()
            meta_json = json.load(fin)
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

    def _load_files(self, idx, folders, suffix):
        '''
        Load corresponding files given index, folders name and file suffix
        The output would be a nested list of nframes * nfolders data
        '''
        if isinstance(idx, int):
            seq_id, frame_id = self._locate_frame(idx)
        else:
            seq_id, frame_id = idx

        ar = zipfile.ZipFile(self.base_path / (seq_id + ".zip"))
        return [[ar.read("%s/%04d.%s" % (f, fidx, suffix)) for f in folders]
            for fidx in range(frame_id - self.nframes, frame_id+1)]

    def lidar_data(self, idx, names=None, concat=False):
        """
        :param names: frame names of lidar to be loaded
        :param concat: concatenate the points together. If concatenated, point cloud will be in vehicle frame (FLU)

        XXX: support return ri2 data
        """
        unpack_result, names = check_frames(names, self.VALID_LIDAR_NAMES)

        outputs = []

        for fdata in self._load_files(idx, names, "npy"):
            clouds = [np.load(BytesIO(buf)) for buf in fdata]

            if concat:
                outputs.append(np.vstack(clouds))
            else:
                calib = self.calibration_data(idx)
                for i, name in enumerate(names):
                    rt = calib.extrinsics[name]
                    clouds[i][:,:3] = clouds[i][:,:3].dot(rt[:3,:3].T) + rt[:3, 3]

                if unpack_result:
                    outputs.append(clouds[0])
                else:
                    outputs.append(clouds)

        if concat or unpack_result:
            return outputs[0] if self.nframes == 0 else outputs
        elif self.nframes == 0:
            return outputs[0]
        else:
            return list(zip(*outputs))

    def camera_data(self, idx, names=None):
        """
        :param names: frame names of camera to be loaded
        """
        unpack_result, names = check_frames(names, self.VALID_CAM_NAMES)

        buffers = self._load_files(idx, names, "jpg")
        outputs = [[Image.open(BytesIO(buf)).convert('RGB') for buf in fdata] for fdata in buffers]
        outputs = list(zip(*outputs)) # transpose nframes * nnames data to nnames * nframes data

        if self.nframes == 0:
            outputs = [d[0] for d in outputs]
        return outputs[0] if unpack_result else outputs

    def camera_label(self, idx, names=None):
        unpack_result, names = check_frames(names, self.VALID_CAM_NAMES)

        buffers = self._load_files(idx, ["label_" + s for s in names], "json")
        outputs = [[list(map(edict, json.loads(buf.decode()))) for buf in fdata] for fdata in buffers]
        outputs = list(zip(*outputs))

        if self.nframes == 0:
            outputs = [d[0] for d in outputs]
        return outputs[0] if unpack_result else outputs

    def lidar_objects(self, idx, raw=False):
        buffers = self._load_files(idx, ["label_lidars"], "json")

        outputs = []
        for din in buffers:
            labels = list(map(edict, json.loads(din[0].decode())))

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
            outputs.append(arr)

        return outputs[0] if self.nframes == 0 else outputs

    def calibration_data(self, idx):
        fname, _ = self._locate_frame(idx)
        calib_params = TransformSet("vehicle")
        ar = zipfile.ZipFile(self.base_path / (fname + ".zip"))

        # load camera calibration
        with ar.open("context/calib_cams.json") as fin:
            calib_cams = json.loads(fin.read().decode())
            for frame, calib in calib_cams.items():
                frame = "camera_" + frame
                (fu, fv, cu, cv), distort = calib['intrinsic'][:4], calib['intrinsic'][4:]
                transform = np.array(calib['extrinsic']).reshape(4,4)
                size = (calib['width'], calib['height'])
                calib_params.set_intrinsic_pinhole(frame, size, cu, cv, fu, fv, distort_coeffs=distort)
                calib_params.set_extrinsic(transform, frame_from=frame)

        # load lidar calibration
        with ar.open("context/calib_lidars.json") as fin:
            calib_lidars = json.loads(fin.read().decode())
            for frame, calib in calib_lidars.items():
                frame = "lidar_" + frame
                calib_params.set_intrinsic_lidar(frame)
                transform = np.array(calib['extrinsic']).reshape(4,4)
                calib_params.set_extrinsic(transform, frame_from=frame)

        return calib_params

    def identity(self, idx):
        fname, fidx = self._locate_frame(idx)
        return self.phase, fname, fidx

    def timestamp(self, idx):
        return [int(d.decode()) for d in self._load_files(idx, "timestamp", "txt")]

def dump_detection_output(detections: Target3DArray, context: str, timestamp: int):
    '''
    :param detections: detection result
    :param ids: auxiliary information for output, each item contains context name and timestamp
    '''
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2

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
        waymo_target.frame_timestamp_micros = timestamp
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

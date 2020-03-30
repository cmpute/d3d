import itertools
import json
import logging
import os
import os.path as osp
import zipfile
from collections import OrderedDict
from enum import Enum
from io import BytesIO

import numpy as np
from addict import Dict as edict
from PIL import Image
from scipy.spatial.transform import Rotation

from d3d.abstraction import ObjectTarget3D, ObjectTarget3DArray, ObjectTag

_logger = logging.getLogger("d3d")

class WaymoObjectClass(Enum):
    Unknown = 0
    Vehicle = 1
    Pedestrian = 2
    Sign = 3
    Cyclist = 4

class Loader:
    VALID_CAM_NAMES = ["front", "front_left", "front_right", "side_left", "side_right"]
    VALID_LIDAR_NAMES = ["top", "front", "side_left", "side_right", "rear"]

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
    def __init__(self, base_path, phase="training"):
        """
        :param phase: training, validation or testing
        """

        self.base_path = osp.join(base_path, phase)
        self._load_metadata()

    def _load_metadata(self):
        self._metadata = OrderedDict()
        for archive in os.listdir(self.base_path):
            name, fext = osp.splitext(archive)
            if fext != ".zip":
                continue

            with zipfile.ZipFile(osp.join(self.base_path, archive)) as ar:
                with ar.open("context/stats.json") as fin:
                    self._metadata[name] = edict(json.loads(fin.read().decode()))

    def __len__(self):
        return sum(v.frame_count for v in self._metadata.values())

    def _locate_frame(self, idx):
        for k, v in self._metadata.items():
            if idx < v.frame_count:
                return k + ".zip", idx
            idx -= v

    def lidar_data(self, idx, names=None, concat=True):
        """
        :param names: names of lidar to be loaded, options: top, front, side_left, side_right, rear
        :param concat: concatenate the points together
        """
        if names is None:
            names = Loader.VALID_LIDAR_NAMES
        else: # sanity check
            for name in names:
                if name not in Loader.VALID_LIDAR_NAMES:
                    raise ValueError("Invalid lidar name, options are top, front, side_left, side_right, rear")

        outputs = []
        fname, fidx = self._locate_frame(idx)
        with zipfile.ZipFile(osp.join(self.base_path, fname)) as ar:
            for name in names:
                with ar.open("lidar_%s/%04d.npy" % (name, fidx)) as fin:
                    buffer = BytesIO(fin.read())
                    outputs.append(np.load(buffer))

        if concat:
            outputs = np.vstack(outputs)
        return outputs

    def camera_data(self, idx, names=None):
        """
        :param names: names of camera to be loaded, options: front, front_left, front_right, side_left, side_right, rear
        """
        if names is None:
            names = Loader.VALID_CAM_NAMES
        else: # sanity check
            for name in names:
                if name not in Loader.VALID_CAM_NAMES:
                    raise ValueError("Invalid camera name, options are top, front, side_left, side_right, rear")

        outputs = []
        fname, fidx = self._locate_frame(idx)
        with zipfile.ZipFile(osp.join(self.base_path, fname)) as ar:
            for name in names:
                with ar.open("camera_%s/%04d.jpg" % (name, fidx)) as fin:
                    outputs.append(Image.open(fin).convert('RGB'))

        return outputs

    def lidar_label(self, idx):
        fname, fidx = self._locate_frame(idx)
        with zipfile.ZipFile(osp.join(self.base_path, fname)) as ar:
            with ar.open("label_lidars/%04d.json" % fidx) as fin:
                return [edict(item) for item in json.loads(fin.read().decode())]

    def camera_label(self, idx, names=None):
        if names is None:
            names = Loader.VALID_CAM_NAMES
        else: # sanity check
            for name in names:
                if name not in Loader.VALID_CAM_NAMES:
                    raise ValueError("Invalid camera name, options are top, front, side_left, side_right, rear")

        outputs = []
        fname, fidx = self._locate_frame(idx)
        with zipfile.ZipFile(osp.join(self.base_path, fname)) as ar:
            for name in names:
                with ar.open("label_camera_%s/%04d.json" % (name, fidx)) as fin:
                    objects = [edict(item) for item in json.loads(fin.read().decode())]
                    outputs.append(objects)

        return outputs

    def lidar_objects(self, idx):
        labels = self.lidar_label(idx)
        outputs = ObjectTarget3DArray()

        for label in labels:
            target = ObjectTarget3D(
                label.center,
                Rotation.from_euler("z", label.heading),
                label.size,
                ObjectTag(label.label, WaymoObjectClass),
                id=label.id
            )
            outputs.append(target)

        return outputs

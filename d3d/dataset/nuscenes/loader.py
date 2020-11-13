import itertools
import json
import logging
import os
import zipfile
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
from addict import Dict as edict
from PIL import Image
from enum import Enum, IntFlag, auto
from scipy.spatial.transform import Rotation

from d3d.abstraction import (ObjectTag, ObjectTarget3D, Target3DArray,
                             TransformSet)
from d3d.dataset.base import DetectionDatasetBase, check_frames, split_trainval
from d3d.dataset.zip import PatchedZipFile

_logger = logging.getLogger("d3d")

class NuscenesObjectClass(IntFlag):
    '''
    Categories and attributes of an annotation in nuscenes

    Encoded into 4bytes integer
    XXXF: level0 category
    XXFX: level1 category
    XFXX: level2 category
    FXXX: attribute
    '''
    unknown = 0x0000

    # categories
    animal = 0x0001
    human = 0x0002
    human_pedestrian = 0x0012
    human_pedestrian_adult = 0x0112
    human_pedestrian_child = 0x0212
    human_pedestrian_construction_worker = 0x0312
    human_pedestrian_personal_mobility = 0x0412
    human_pedestrian_police_officer = 0x0512
    human_pedestrian_stroller = 0x0612
    human_pedestrian_wheelchair = 0x0712
    movable_object = 0x0003
    movable_object_barrier = 0x0013
    movable_object_debris = 0x0023
    movable_object_pushable_pullable = 0x0033
    movable_object_trafficcone = 0x0043
    vehicle_bicycle = 0x0004
    vehicle_bus = 0x0014
    vehicle_bus_bendy = 0x0114
    vehicle_bus_rigid = 0x0214
    vehicle_car = 0x0024
    vehicle_construction = 0x0034
    vehicle_emergency = 0x0044
    vehicle_emergency_ambulance = 0x0144
    vehicle_emergency_police = 0x0244
    vehicle_motorcycle = 0x0054
    vehicle_trailer = 0x0064
    vehicle_truck = 0x0074
    static_object = 0x0005
    static_object_bicycle_rack = 0x0015

    # attributes
    vehicle_moving = 0x1000
    vehicle_stopped = 0x2000
    vehicle_parked = 0x3000
    cycle_with_rider = 0x4000
    cycle_without_rider = 0x5000
    pedestrian_sitting_lying_down = 0x6000
    pedestrian_standing = 0x7000
    pedestrian_moving = 0x8000

    @classmethod
    def parse(cls, string):
        return cls[string.replace('.', '_')]

    @property
    def category(self):
        return self & 0x0fff
    @property
    def attribute(self):
        return self & 0xf000
    @property
    def category_name(self):
        name = self.category.name
        name = name.replace("icle_", "icle.").replace("an_", "an.")
        name = name.replace("t_", "t.").replace("s_", "s.")
        name = name.replace("y_", "y.")
        return name
    @property
    def attribute_name(self):
        name = self.attribute.name
        name = name.replace("e_", "e.")
        name = name.replace("n_", "n.")
        return name
    @property
    def pretty_name(self):
        return f"{self.category_name}[{self.attribute_name}]"

    def to_detection(self):
        """
        Convert the class to class for detection
        """
        # following table is copied from nuscenes definition
        detection_mapping = {
            'movable_object.barrier': 'barrier',
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'movable_object.trafficcone': 'traffic_cone',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck'
        }

        if self.category_name not in detection_mapping:
            return NuscenesDetectionClass.ignore
        else:
            return NuscenesDetectionClass[detection_mapping[self.category_name]]

class NuscenesDetectionClass(Enum):
    ignore = 0
    barrier = auto()
    bicycle = auto()
    bus = auto()
    car = auto()
    construction_vehicle = auto()
    motorcycle = auto()
    pedestrian = auto()
    traffic_cone = auto()
    trailer = auto()
    truck = auto()

class NuscenesObjectLoader(DetectionDatasetBase): # TODO(v0.4): rename to NuscenesLoader
    '''
    Load waymo dataset into a usable format.
    Please use the d3d_nuscenes_convert command (do not use --all-frames) to convert the dataset first into following formats

    # Directory Structure
    - <base_path directory>
        - trainval
            - scene_xxx.zip
            - ...
        - test
            - scene_xxx.zip
            - ...
    '''
    VALID_CAM_NAMES = ["cam_front", "cam_front_left", "cam_front_right", "cam_back", "cam_back_left", "cam_back_right"]
    VALID_LIDAR_NAMES = ["lidar_top"]

    def __init__(self, base_path, inzip=False, phase="training", trainval_split=1, trainval_random=False):
        """
        :param phase: training, validation or testing
        :param trainval_split: placeholder for interface compatibility with other loaders
        """

        if not inzip:
            raise NotImplementedError("Currently only support load from zip files in Nuscenes dataset")
        if phase not in ['training', 'validation', 'testing']:
            raise ValueError("Invalid phase tag")

        self.base_path = Path(base_path) / ("trainval" if phase in ["training", "validation"] else "test")
        self.phase = phase
        self._load_metadata()

        # split trainval
        total_count = sum(v.nbr_samples for v in self._metadata.values())
        self.frames = split_trainval(phase, total_count, trainval_split, trainval_random)

    def _load_metadata(self):
        meta_path = self.base_path / "metadata.json"
        if not meta_path.exists():
            _logger.info("Creating metadata of Nuscenes dataset (%s)...", self.phase)
            metadata = {}

            for archive in self.base_path.iterdir():
                if archive.is_dir() or archive.suffix != ".zip":
                    continue

                with zipfile.ZipFile(archive) as ar:
                    with ar.open("scene/stats.json") as fin:
                        metadata[archive.stem] = json.loads(fin.read().decode())
            with open(meta_path, "w") as fout:
                json.dump(metadata, fout)

        with open(meta_path) as fin:
            self._metadata = OrderedDict()
            meta_json = json.load(fin)
            for k, v in meta_json.items():
                self._metadata[k] = edict(v)

    def __len__(self):
        return len(self.frames)

    def _locate_frame(self, idx):
        # use underlying frame index
        idx = self.frames[idx]

        # find corresponding sample
        for k, v in self._metadata.items():
            if idx < v.nbr_samples:
                return k, idx
            idx -= v.nbr_samples
        raise ValueError("Index larger than dataset size")

    def _locate_file(self, idx, folders, suffix):
        fname, fidx = self._locate_frame(idx)
        if isinstance(folders, list):
            names = ["%s/%03d.%s" % (f, fidx, suffix) for f in folders]
            ar = PatchedZipFile(self.base_path / (fname + ".zip"), to_extract=names)
            return [ar.open(n) for n in names]
        else:
            name = "%s/%03d.%s" % (folders, fidx, suffix)
            ar = PatchedZipFile(self.base_path / (fname + ".zip"), to_extract=name)
            return ar.open(name)

    def map_data(self, idx):
        # XXX: see https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/ for image cropping
        raise NotImplementedError()

    def lidar_data(self, idx, names='lidar_top', concat=False):
        if isinstance(names, str):
            names = [names]
        if names != self.VALID_LIDAR_NAMES:
            raise ValueError("There's only one lidar in Nuscenes dataset")

        with self._locate_file(idx, "lidar_top", "pcd") as fin:
            buffer = fin.read()
        scan = np.frombuffer(buffer, dtype=np.float32)
        scan = np.copy(scan.reshape(-1, 5)) # (x, y, z, intensity, ring index)

        if concat: # convert lidar to base frame
            calib = self.calibration_data(idx)
            rt = np.linalg.inv(calib.extrinsics[names[0]])
            scan[:,:3] = scan[:,:3].dot(rt[:3,:3].T) + rt[:3, 3]

        return scan

    def camera_data(self, idx, names=None):
        unpack_result, names = check_frames(names, self.VALID_CAM_NAMES)

        handles = self._locate_file(idx, names, "jpg")
        outputs = [Image.open(h).convert('RGB') for h in handles]
        map(lambda h: h.close(), handles)

        if unpack_result:
            return outputs[0]
        else:
            return outputs

    def lidar_objects(self, idx, raw=False, convert_tag=False):        
        with self._locate_file(idx, "annotation", "json") as fin:
            labels = list(map(edict, json.loads(fin.read().decode())))

        if raw:
            return labels

        ego_r, ego_t = self.pose(idx)
        outputs = Target3DArray(frame="ego")
        for label in labels:
            # convert tags
            tag = NuscenesObjectClass.parse(label.category)
            for attr in label.attribute:
                tag = tag | NuscenesObjectClass.parse(attr)
            if convert_tag:
                tag = ObjectTag(tag.to_detection(), NuscenesDetectionClass)
            else:
                tag = ObjectTag(tag, NuscenesObjectClass)

            # caculate relative pose
            r = Rotation.from_quat(label.rotation[1:] + [label.rotation[0]])
            t = label.translation
            rel_r = ego_r.inv() * r
            rel_t = np.dot(ego_r.inv().as_matrix(), t - ego_t)
            size = [label.size[1], label.size[0], label.size[2]] # wlh -> lwh

            # create object
            tid = int(label.instance[:8], 16) # truncate into uint64
            target = ObjectTarget3D(rel_t, rel_r, size, tag, tid=tid)
            outputs.append(target)

        return outputs

    def calibration_data(self, idx):
        fname, _ = self._locate_frame(idx)
        calib_params = TransformSet("ego")
        ar = PatchedZipFile(self.base_path / (fname + ".zip"), to_extract="scene/calib.json")

        with ar.open("scene/calib.json") as fin:
            calib_data = json.loads(fin.read().decode())
            for frame, calib in calib_data.items():
                # set intrinsics
                if frame.startswith('cam'):
                    image_size = (1600, 900) # currently all the images have the same size
                    projection = np.array(calib['camera_intrinsic'])
                    calib_params.set_intrinsic_camera(frame, projection, image_size, rotate=False)
                elif frame.startswith('lidar'):
                    calib_params.set_intrinsic_lidar(frame)
                elif frame.startswith('radar'):
                    calib_params.set_intrinsic_radar(frame)
                else:
                    raise ValueError("Unrecognized frame name.")

                # set extrinsics
                r = Rotation.from_quat(calib['rotation'][1:] + [calib['rotation'][0]])
                t = np.array(calib['translation'])
                extri = np.eye(4)
                extri[:3, :3] = r.as_matrix()
                extri[:3, 3] = t
                calib_params.set_extrinsic(extri, frame_from=frame)

        return calib_params

    def identity(self, idx):
        scene, fidx = self._locate_frame(idx)
        return self.phase, scene, fidx

    def timestamp(self, idx):
        with self._locate_file(idx, "timestamp", "txt") as fin:
            return int(fin.read())

    def pose(self, idx):
        '''
        Return (rotation, translation)
        '''
        with self._locate_file(idx, "pose", "json") as fin:
            data = json.loads(fin.read().decode())
        r = Rotation.from_quat(data['rotation'][1:] + [data['rotation'][0]])
        t = np.array(data['translation'])
        return r, t

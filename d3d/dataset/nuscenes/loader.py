import itertools
import json
import logging
import os
import zipfile
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
import msgpack
from addict import Dict as edict
from PIL import Image
from enum import Enum, IntFlag, auto
from scipy.spatial.transform import Rotation

from d3d.abstraction import (ObjectTag, ObjectTarget3D, Target3DArray,
                             TrackingTarget3D, TransformSet, EgoPose)
from d3d.dataset.base import TrackingDatasetBase, check_frames, split_trainval, expand_idx, expand_idx_name
from d3d.dataset.zip import PatchedZipFile

_logger = logging.getLogger("d3d")

class NuscenesObjectClass(IntFlag):
    '''
    Categories and attributes of an annotation in nuscenes

    Encoded into 4bytes integer
    0xFFFF
      │││└: level0 category
      ││└─: level1 category
      │└──: level2 category
      └───: attribute
    '''
    unknown = 0x0000
    noise = 0x0010

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
    vehicle_ego = 0x0084
    static_object = 0x0005
    static_object_bicycle_rack = 0x0015
    flat = 0x0006
    flat_driveable_surface = 0x0016
    flat_sidewalk = 0x0026
    flat_terrain = 0x0036
    flat_other = 0x0046
    static = 0x0007
    static_manmade = 0x0017
    static_vegetation = 0x0027
    static_other = 0x0037

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

class NuscenesLoader(TrackingDatasetBase):
    '''
    Load Nuscenes dataset into a usable format.
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
    VALID_OBJ_CLASSES = NuscenesDetectionClass

    def __init__(self, base_path, inzip=False, phase="training", trainval_split=1, trainval_random=False, nframes=0):
        """
        :param phase: training, validation or testing
        :param trainval_split: placeholder for interface compatibility with other loaders
        """
        super().__init__(base_path, inzip=inzip, phase=phase, nframes=nframes,
                         trainval_split=trainval_split, trainval_random=trainval_random)

        self.base_path = Path(base_path) / ("trainval" if phase in ["training", "validation"] else "test")
        self.inzip = inzip

        self._metadata = None
        self._segmapping = None
        self._load_metadata()

        # split trainval
        total_count = sum(v.nbr_samples for v in self._metadata.values()) - nframes * len(self._metadata)
        self.frames = split_trainval(phase, total_count, trainval_split, trainval_random)

    def _load_metadata(self):
        meta_path = self.base_path / "metadata.msg"
        if not meta_path.exists():
            _logger.info("Creating metadata of Nuscenes dataset (%s)...", self.phase)
            metadata = {}

            if self.inzip:
                for archive in self.base_path.iterdir():
                    if archive.is_dir() or archive.suffix != ".zip":
                        continue

                    with PatchedZipFile(archive, to_extract="scene/stats.json") as ar:
                        metadata[archive.stem] = json.loads(ar.read("scene/stats.json"))
            else:
                for folder in self.base_path.iterdir():
                    if not folder.is_dir() or folder.name == "maps":
                        continue

                    metadata[folder.name] = json.loads((folder / "scene/stats.json").read_text())

            assert len(metadata) > 0, "The dataset folder contains no valid frame, "\
                "please check path or parameters!"
            with open(meta_path, "wb") as fout:
                msgpack.pack(metadata, fout)

        with open(meta_path, "rb") as fin:
            self._metadata = OrderedDict()
            meta_json = msgpack.unpack(fin)
            for k, v in meta_json.items():
                self._metadata[k] = edict(v)

        # load category mapping for segmentation
        cat_path = self.base_path / "lidarseg_category.json"
        if cat_path.exists():
            with open(cat_path) as fin:
                cat_json = json.load(fin)
            cat_dict = {}
            for item in cat_json:
                cat_dict[item['index']] = NuscenesObjectClass.parse(item['name'])
            
            self._segmapping = np.empty(max(cat_dict.keys()) + 1, dtype='u4')
            for idx, clsobj in cat_dict.items():
                self._segmapping[idx] = clsobj.value

    def __len__(self):
        return len(self.frames)

    @property
    def sequence_ids(self):
        return list(self._metadata.keys())

    @property
    def sequence_sizes(self):
        return {k: v.nbr_samples for k, v in self._metadata.items()}

    def _locate_frame(self, idx):
        # use underlying frame index
        idx = self.frames[idx]

        # find corresponding sample
        for k, v in self._metadata.items():
            if idx < (v.nbr_samples - self.nframes):
                return k, idx
            idx -= (v.nbr_samples - self.nframes)
        raise ValueError("Index larger than dataset size")

    def map_data(self, idx):
        # TODO(v0.5): see https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/ for image cropping
        raise NotImplementedError()

    def _load_lidar_data(self, seq_id, fname):
        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                buffer = ar.read(fname)
        else:
            buffer = (self.base_path / seq_id / fname).read_bytes()

        scan = np.frombuffer(buffer, dtype=np.float32)
        scan = np.copy(scan.reshape(-1, 5)) # (x, y, z, intensity, ring index)
        return scan

    @expand_idx_name(VALID_LIDAR_NAMES)
    def lidar_data(self, idx, names='lidar_top'):
        seq_id, frame_idx = idx
        fname = "lidar_top/%03d.pcd" % frame_idx

        if self._return_file_path:
            return self.base_path / seq_id / fname
        else:
            return self._load_lidar_data(seq_id, fname)

    def _load_camera_data(self, seq_id, fname):
        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                return Image.open(ar.open(fname)).convert('RGB')
        else:
            return Image.open(self.base_path / seq_id / fname)

    @expand_idx_name(VALID_CAM_NAMES)
    def camera_data(self, idx, names=None):
        seq_id, frame_idx = idx
        fname = "%s/%03d.jpg" % (names, frame_idx)

        if self._return_file_path:
            return self.base_path / seq_id / fname
        else:
            return self._load_camera_data(seq_id, fname)

    @expand_idx_name(VALID_CAM_NAMES + VALID_LIDAR_NAMES)
    def intermediate_data(self, idx, names=None, ninter_frames=1):
        seq_id, frame_idx = idx
        fname = "intermediate/%03d/meta.json" % frame_idx

        # Load meta json
        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                meta = json.loads(ar.read(fname))
        else:
            with (self.base_path / seq_id / fname).open() as fin:
                meta = json.load(fin)
        if not meta:
            return []

        # Select frames
        if ninter_frames is None:
            meta = [edict(item) for item in meta[names]]
        else:
            meta = [edict(item) for item in meta[names][:ninter_frames]]

        # parse pose part
        for item in meta:
            rotation = item.pop("rotation")
            rotation = Rotation.from_quat(rotation[1:] + [rotation[0]])
            translation = item.pop("translation")
            translation = rotation.inv().as_matrix().dot(translation)
            item.pose = EgoPose(translation, rotation)

        if self._return_file_path:
            for item in meta:
                item.file = self.base_path / seq_id / "intermediate" / f"{frame_idx:03}" / item.file
            return meta

        # Load actual data
        for item in meta:
            fname = "intermediate/%03d/%s" % (frame_idx, item.pop("file"))
            if names in self.VALID_CAM_NAMES:
                item.data = self._load_camera_data(seq_id, fname)
            else: # names in VALID_LIDAR_NAMES:
                item.data = self._load_lidar_data(seq_id, fname)
        return meta

    @expand_idx
    def annotation_3dobject(self, idx, raw=False, convert_tag=True, with_velocity=True):
        seq_id, frame_idx = idx
        fname = "annotation/%03d.json" % frame_idx
        if self._return_file_path:
            return self.base_path / seq_id / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                labels = json.loads(ar.read(fname))
        else:
            with (self.base_path / seq_id / fname).open() as fin:
                labels = json.load(fin)
        labels = list(map(edict, labels))
        if raw:
            return labels

        # parse annotations
        ego_pose = self.pose(idx, bypass=True)
        ego_r, ego_t = ego_pose.orientation, ego_pose.position
        ego_t = ego_r.as_matrix().dot(ego_t) # convert to original representation
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
            tid = int(label.instance[:8], 16) # truncate into uint64

            # create object
            if with_velocity:
                v = np.dot(ego_r.inv().as_matrix(), label.velocity)
                w = label.angular_velocity
                target = TrackingTarget3D(rel_t, rel_r, size, v, w, tag, tid=tid)
                outputs.append(target)
            else:
                target = ObjectTarget3D(rel_t, rel_r, size, tag, tid=tid)
                outputs.append(target)

        return outputs

    @expand_idx
    def annotation_3dpoints(self, idx, raw=False):
        seq_id, frame_idx = idx

        fname = "lidar_top_seg/%03d.bin" % frame_idx
        if self._return_file_path:
            return self.base_path / seq_id / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                buffer = ar.read(fname)
        else:
            buffer = (self.base_path / seq_id / fname).read_bytes()
        label = np.frombuffer(buffer, dtype='u1')

        if raw:
            return label
        else:
            return self._segmapping[label]

    @expand_idx
    def metadata(self, idx):
        seq_id, frame_idx = idx
        assert not self._return_file_path, "The metadata is not in a single file!"

        meta = self._metadata[seq_id]
        return edict(
            scene_description=meta.description,
            scene_token=meta.token,
            sample_token=meta.sample_tokens[frame_idx],
            logfile=meta.logfile,
            date_captured=meta.date_captured,
            vehicle=meta.vehicle,
            location=meta.location
        )

    @expand_idx
    def calibration_data(self, idx):
        seq_id, _ = idx
        assert not self._return_file_path, "The calibration is not in a single file!"

        calib_params = TransformSet("ego")
        calib_fname = "scene/calib.json"
        if self.inzip:
            with PatchedZipFile(self.base_path / (seq_id + ".zip"), to_extract=calib_fname) as ar:
                calib_data = json.loads(ar.read(calib_fname))
        else:
            with (self.base_path / seq_id / calib_fname).open() as fin:
                calib_data = json.load(fin)

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

    @expand_idx
    def identity(self, idx):
        return idx

    @expand_idx_name(VALID_LIDAR_NAMES + VALID_CAM_NAMES)
    def timestamp(self, idx, names="lidar_top"):
        seq_id, frame_idx = idx
        fname = "timestamp/%03d.json" % frame_idx
        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                tsdict = json.loads(ar.read(fname))
        else:
            with (self.base_path / seq_id / fname).open() as fin:
                tsdict = json.load(fin)
        return tsdict[names]

    @expand_idx_name(VALID_LIDAR_NAMES + VALID_CAM_NAMES)
    def pose(self, idx, names="lidar_top"):
        # Note that here pose always return the pose of the vehicle, names are for different timestamps
        seq_id, frame_idx = idx
        fname = "pose/%03d.json" % frame_idx
        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                data = json.loads(ar.read(fname))
        else:
            with (self.base_path / seq_id / fname).open() as fin:
                data = json.load(fin)

        data = data[names]
        r = Rotation.from_quat(data['rotation'][1:] + [data['rotation'][0]])
        t = r.inv().as_matrix().dot(np.array(data['translation']))
        return EgoPose(t, r)

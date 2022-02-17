from datetime import datetime
import os
import xml.etree.ElementTree as ET
from collections import namedtuple
from enum import Enum, auto
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from d3d.abstraction import EgoPose

# ====== Common classes ======

# see data format description from raw dataset
OxtData = namedtuple("OxtData", [
    'lat',          # latitude of the oxts-unit (deg)
    'lon',          # longitude of the oxts-unit (deg)
    'alt',          # altitude of the oxts-unit (m)
    'roll',         # roll angle (rad),    0 = level, positive = left side up,      range-pi   .. +pi
    'pitch',        # pitch angle (rad),   0 = level, positive = front down,        range-pi/2 .. +pi/2
    'yaw',          # heading (rad),       0 = east,  positive = counter clockwise, range-pi   .. +pi
    'vn',           # velocity towards north (m/s)
    've',           # velocity towards east (m/s)
    'vf',           # forward velocity, i.e. parallel to earth-surface (m/s)
    'vl',           # leftward velocity, i.e. parallel to earth-surface (m/s)
    'vu',           # upward velocity, i.e. perpendicular to earth-surface (m/s)
    'ax',           # acceleration in x, i.e. in direction of vehicle front (m/s^2)
    'ay',           # acceleration in y, i.e. in direction of vehicle left (m/s^2)
    'az',           # acceleration in z, i.e. in direction of vehicle top (m/s^2)
    'af',           # forward acceleration (m/s^2)
    'al',           # leftward acceleration (m/s^2)
    'au',           # upward acceleration (m/s^2)
    'wx',           # angular rate around x (rad/s)
    'wy',           # angular rate around y (rad/s)
    'wz',           # angular rate around z (rad/s)
    'wf',           # angular rate around forward axis (rad/s)
    'wl',           # angular rate around leftward axis (rad/s)
    'wu',           # angular rate around upward axis (rad/s)
    'pos_accuracy', # position accuracy (north/east in m)
    'vel_accuracy', # velocity accuracy (north/east in m/s)
    'navstat',      # navigation status (see navstat_to_string)
    'numsats',      # number of satellites tracked by primary GPS receiver
    'posmode',      # position mode of primary GPS receiver (see gps_mode_to_string)
    'velmode',      # velocity mode of primary GPS receiver (see gps_mode_to_string)
    'orimode',      # orientation mode of primary GPS receiver (see gps_mode_to_string)
])

class KittiObjectClass(Enum):
    '''
    Category of objects in KITTI dataset
    '''

    DontCare = 0
    Car = auto()
    Van = auto()
    Truck = auto()
    Pedestrian = auto()
    Person = auto() # Person (sitting).
    Person_sitting = Person
    Cyclist = auto()
    Tram = auto()
    Misc = auto()

class SemanticKittiLearningClass(Enum):
    """
    ID from the learning map in the official devkit.
    Reference: https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
    """
    unlabeled = 0
    car = 1
    bicycle = 2
    motorcycle = 3
    truck = 4
    other_vehicle = 5
    person = 6
    bicyclist = 7
    motorcyclist = 8
    road = 9
    parking = 10
    sidewalk = 11
    other_ground = 12
    building = 13
    fence = 14
    vegetation = 15
    trunk = 16
    terrain = 17
    pole = 18
    traffic_sign = 19
    moving_car = 20
    moving_bicyclist = 21
    moving_person = 22
    moving_motorcyclist = 23
    moving_other_vehicle = 24
    moving_truck = 25
    
    def to_original_id(self):
        learning_map_inv = {
            0: 0   ,
            1: 10  ,
            2: 11  ,
            3: 15  ,
            4: 18  ,
            5: 20  ,
            6: 30  ,
            7: 31  ,
            8: 32  ,
            9: 40  ,
            10: 44 ,
            11: 48 ,
            12: 49 ,
            13: 50 ,
            14: 51 ,
            15: 70 ,
            16: 71 ,
            17: 72 ,
            18: 80 ,
            19: 81 ,
            20: 252,
            21: 253,
            22: 254,
            23: 255,
            24: 259,
            25: 258,
        }
        return SemanticKittiClass(learning_map_inv[self.value])

class SemanticKittiClass(Enum):
    unlabeled = 0
    outlier = 1
    # static = 9
    car = 10
    bicycle = 11
    bus = 13
    motorcycle = 15
    on_rails = 16
    truck = 18
    other_vehicle = 20
    person = 30
    bicyclist = 31
    motorcyclist = 32
    road = 40
    parking = 44
    sidewalk = 48
    other_ground = 49
    building = 50
    fence = 51
    other_structure = 52
    lane_marking = 60
    vegetation = 70
    trunk = 71
    terrain = 72
    pole = 80
    traffic_sign = 81
    other_object = 99
    moving_car = 252
    moving_bicyclist = 253
    moving_person = 254
    moving_motorcyclist = 255
    moving_on_rails = 256
    moving_bus = 257
    moving_truck = 258
    moving_other_vehicle = 259

    @property
    def color(self):
        color_map = { # bgr
            0 : [0, 0, 0],
            1 : [0, 0, 255],
            9 : [0, 0, 0],
            10: [245, 150, 100],
            11: [245, 230, 100],
            13: [250, 80, 100],
            15: [150, 60, 30],
            16: [255, 0, 0],
            18: [180, 30, 80],
            20: [255, 0, 0],
            30: [30, 30, 255],
            31: [200, 40, 255],
            32: [90, 30, 150],
            40: [255, 0, 255],
            44: [255, 150, 255],
            48: [75, 0, 75],
            49: [75, 0, 175],
            50: [0, 200, 255],
            51: [50, 120, 255],
            52: [0, 150, 255],
            60: [170, 255, 150],
            70: [0, 175, 0],
            71: [0, 60, 135],
            72: [80, 240, 150],
            80: [150, 240, 255],
            81: [0, 0, 255],
            99: [255, 255, 50],
            252: [245, 150, 100],
            256: [255, 0, 0],
            253: [200, 40, 255],
            254: [30, 30, 255],
            255: [90, 30, 150],
            257: [250, 80, 100],
            258: [180, 30, 80],
            259: [255, 0, 0],
        }
        b, g, r = color_map[self.value]
        return r, g, b

    def to_learning_id(self, static_only=True):
        learning_map = {
            0  : 0  ,  # "unlabeled"
            1  : 0  ,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10 : 1  ,  # "car"
            11 : 2  ,  # "bicycle"
            13 : 5  ,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15 : 3  ,  # "motorcycle"
            16 : 5  ,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18 : 4  ,  # "truck"
            20 : 5  ,  # "other-vehicle"
            30 : 6  ,  # "person"
            31 : 7  ,  # "bicyclist"
            32 : 8  ,  # "motorcyclist"
            40 : 9  ,  # "road"
            44 : 10 ,  # "parking"
            48 : 11 ,  # "sidewalk"
            49 : 12 ,  # "other-ground"
            50 : 13 ,  # "building"
            51 : 14 ,  # "fence"
            52 : 0  ,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60 : 9  ,  # "lane-marking" to "road" ---------------------------------mapped
            70 : 15 ,  # "vegetation"
            71 : 16 ,  # "trunk"
            72 : 17 ,  # "terrain"
            80 : 18 ,  # "pole"
            81 : 19 ,  # "traffic-sign"
            99 : 0  ,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 1 if static_only else 20,  # "moving-car"
            253: 7 if static_only else 21,  # "moving-bicyclist"
            254: 6 if static_only else 22,  # "moving-person"
            255: 8 if static_only else 23,  # "moving-motorcyclist"
            256: 5 if static_only else 24,  # "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
            257: 5 if static_only else 24,  # "moving-bus" mapped to "moving-other-vehicle" -----------mapped
            258: 4 if static_only else 25,  # "moving-truck"
            259: 5 if static_only else 24,  # "moving-other-vehicle"
        }
        return SemanticKittiLearningClass(learning_map[self.value])


# ========== Loaders ==========


def load_timestamps(basepath, file, formatted=False):
    """
    Read in timestamp file and parse to a list
    """
    timestamps = []
    if isinstance(basepath, (str, Path)):
        fin = Path(basepath, file).open()
    else:  # assume ZipFile object
        fin = basepath.open(str(file))

    with fin:
        if formatted:
            tz_offset = np.timedelta64(1, 'h') # convert German time to UTC time
            for line in fin.readlines():
                timestamps.append(np.datetime64(line.strip()) - tz_offset)
            timestamps = np.asanyarray(timestamps)
        else:
            timestamps = (np.loadtxt(fin) * 1e9).astype("M8[ns]")

    return timestamps


def load_calib_file(basepath, file):
    """
    Read in a calibration file and parse into a dictionary.
    Accept path or file object as input
    """
    data = {}
    if isinstance(basepath, (str, Path)):
        fin = Path(basepath, file).open()
    else:  # assume ZipFile object
        fin = basepath.open(str(file))

    with fin:
        for line in fin.readlines():
            if not line.strip():
                continue
            if not isinstance(line, str):
                line = line.decode()

            if ':' in line:
                key, value = line.split(':', 1)
            else:
                key, value = line.split(" ", 1)
            # The only non-float values in these files are dates, which we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def load_oxt_file(basepath, file):
    data = []
    if isinstance(basepath, (str, Path)):
        fin = Path(basepath, file).open()
    else:  # assume ZipFile object
        fin = basepath.open(str(file))

    with fin:
        for line in fin.readlines():
            if not line.strip():
                continue
            if not isinstance(line, str):
                line = line.decode()

            values = list(map(float, line.strip().split(' ')))
            values[-5:] = list(map(int, values[-5:]))
            data.append(OxtData(*values))

    return data
    
def parse_pose_from_oxt(oxt: OxtData) -> EgoPose:
    import utm
    x, y, *_ = utm.from_latlon(oxt.lat, oxt.lon)
    t = [x, y, oxt.alt]
    r = Rotation.from_euler("xyz", [oxt.roll, oxt.pitch, oxt.yaw])
    return EgoPose(t, r, position_var=np.eye(3) * oxt.pos_accuracy)


def load_image(basepath, file, gray=False):
    """Load an image from file. Accept path or file object as basepath"""
    if isinstance(basepath, (str, Path)):
        return Image.open(Path(basepath, file)).convert('L' if gray else 'RGB')
    else:  # assume ZipFile object
        return Image.open(basepath.open(str(file))).convert('L' if gray else 'RGB')


def load_velo_scan(basepath, file, formatted=False):
    """Load and parse a kitti point cloud file. Accept path or file object as basepath"""
    if isinstance(basepath, (str, Path)):
        scan = np.fromfile(Path(basepath, file), dtype=np.float32)
    else:
        with basepath.open(str(file)) as fin:
            buffer = fin.read()
        scan = np.frombuffer(buffer, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    if not formatted:
        return scan
    columns = ['x', 'y', 'z', 'intensity']
    return scan.view([(c, 'f4') for c in columns])

# ===== Raw data tracklets =====
class _TrackletPose(object):
    def __init__(self, xmlnode):
        for prop in xmlnode:
            setattr(self, prop.tag, float(prop.text))


class _TrackletObject(object):
    def __init__(self, xmlnode):
        for prop in xmlnode:
            if prop.tag == 'poses':
                self.poses = [_TrackletPose(item)
                              for item in prop if item.tag == 'item']
            elif prop.tag == "objectType":
                self.objectType = prop.text
            else:
                setattr(self, prop.tag, float(prop.text))


def load_tracklets(basepath, file):
    if isinstance(basepath, (str, Path)):
        fin = Path(basepath, file).open()
    else:  # assume ZipFile object
        fin = basepath.open(str(file))

    with fin:
        root = ET.fromstring(fin.read())
        root_tracklet = next(iter(root))
        tracklets = [_TrackletObject(item)
                     for item in root_tracklet if item.tag == 'item']
        return tracklets

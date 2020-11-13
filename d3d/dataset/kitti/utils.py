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
    'roll',         # roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
    'pitch',        # pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
    'yaw',          # heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
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

    tz_offset = np.timedelta64(1,'h') # convert German time to UTC time
    with fin:
        if formatted:
            for line in fin.readlines():
                timestamps.append(np.datetime64(line) - tz_offset)
            timestamps = np.asanyarray(timestamps)
        else:
            timestamps = (np.loadtxt(fin) * 1e9).astype("M8[ns]") - tz_offset

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
    r = Rotation.from_euler("xyz", [oxt.roll, oxt.pitch, oxt.yaw + np.pi/2])
    return EgoPose(t, r, position_var=np.eye(3) * oxt.pos_accuracy)


def load_image(basepath, file, gray=False):
    """Load an image from file. Accept path or file object as basepath"""
    if isinstance(basepath, (str, Path)):
        return Image.open(Path(basepath, file)).convert('L' if gray else 'RGB')
    else:  # assume ZipFile object
        return Image.open(basepath.open(str(file))).convert('L' if gray else 'RGB')


def load_velo_scan(basepath, file, binary=True):
    """Load and parse a kitti point cloud file. Accept path or file object as basepath"""
    if binary:
        if isinstance(basepath, (str, Path)):
            scan = np.fromfile(Path(basepath, file), dtype=np.float32)
        else:
            with basepath.open(str(file)) as fin:
                buffer = fin.read()
            scan = np.frombuffer(buffer, dtype=np.float32)
    else:
        if isinstance(basepath, (str, Path)):
            scan = np.loadtxt(Path(basepath, file), dtype=np.float32)
        else:
            scan = np.loadtxt(basepath.open(str(file)), dtype=np.float32)
    return scan.reshape((-1, 4))

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

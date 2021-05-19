from collections import namedtuple
from enum import IntFlag
from pathlib import Path
from typing import List

import numpy as np
from addict import Dict as edict
from d3d.abstraction import EgoPose, ObjectTag, ObjectTarget3D, Target3DArray
from d3d.dataset.kitti.utils import load_image, load_velo_scan
from scipy.spatial.transform import Rotation

INSPVAX = namedtuple("INSPVAX", [ # INSPVAX message from novatel
    "latitude", # (degrees)
    "longitude", # (degrees)
    "altitude", # (m)
    "undulation", # (m)
    "latitude_std", # (m)
    "longitude_std", # (m)
    "altitude_std", # (m)
    "roll", # (degrees)
    "pitch", # (degrees)
    "azimuth", # (degrees)
    "roll_std", # (degrees)
    "pitch_std", # (degrees)
    "azimuth_std", # (degrees)
    "ins_status", #
    "position_type", #
    "extended_status", #
    "seconds_since_update", # (seconds)
    "north_velocity", # (m/s)
    "east_velocity", # (m/s)
    "up_velocity", # (m/s)
    "north_velocity_std", # (m/s)
    "east_velocity_std", # (m/s)
    "up_velocity_std", # (m/s)
])

class CADCObjectClass(IntFlag):
    '''
    Category of objects in CADC dataset

    The ids are encoded into 3bytes integer::

        0x0FFF
           ││└: level0 category (label)
           │└─: level1 category (attribute)
           └──: state
    '''

    Unknown = 0
    Car = 0x0001

    Truck = 0x0002
    Snowplow_Truck = 0x0012
    Semi_Truck = 0x0022
    Construction_Truck = 0x0032
    Garbage_Truck = 0x0042
    Pickup_Truck = 0x0052
    Emergency_Truck = 0x0062

    Bus = 0x0003
    Coach_Bus = 0x0013
    Transit_Bus = 0x0023
    Standard_School_Bus = 0x0033
    Van_School_Bus = 0x0043

    Bicycle = 0x0004
    With_Rider = 0x0014
    Without_Rider = 0x0024

    Horse_and_Buggy = 0x0005
    Pedestrian = 0x0006
    Pedestrian_With_Object = 0x0007
    Animal = 0x0008
    Garbage_Containers_on_Wheels = 0x0009
    Traffic_Guidance_Objects = 0x0010

    # states
    Parked = 0x0100
    Stopped = 0x0200
    Moving = 0x0300

def load_inspvax(basepath, file, labeled=True) -> INSPVAX:
    if isinstance(basepath, (str, Path)):
        data = Path(basepath, file).read_bytes()
    else:  # assume ZipFile object
        data = basepath.read(str(file))

    values = list(map(float, data.strip().split(b' ')))
    if labeled:
        values[13:14] = list(map(int, values[13:14]))
        values.extend([float('nan')] * 8)
    else:
        values[13:16] = list(map(int, values[13:16]))
    return INSPVAX(*values)

def parse_pose_from_inspvax(data: INSPVAX) -> EgoPose:
    import utm
    y, x, *_ = utm.from_latlon(data.latitude, data.longitude)
    t = [x, y, data.altitude]
    r = Rotation.from_euler("xyz", [data.roll, data.pitch, data.azimuth])
    return EgoPose(t, r,
                   position_var=np.asarray([data.latitude_std, data.longitude_std, data.altitude_std]),
                   orientation_var=np.asarray([data.roll_std, data.pitch_std, data.azimuth_std]))

def load_timestamps(basepath, file) -> np.array:
    """
    Read in timestamp file and parse to a list
    """
    timestamps = []
    if isinstance(basepath, (str, Path)):
        fin = Path(basepath, file).open()
    else:  # assume ZipFile object
        fin = basepath.open(str(file))

    tz_offset = np.timedelta64(-4, 'h') # convert German time to UTC time
    with fin:
        for line in fin.readlines():
            timestamps.append(np.datetime64(line.strip()) - tz_offset)

    return np.asanyarray(timestamps)

def load_3d_ann(ditem: list) -> Target3DArray:
    obj_arr = Target3DArray(frame="lidar")
    for box in ditem['cuboids']:
        box = edict(box)
        if box.attributes.truck_type:
            label = CADCObjectClass[box.attributes.truck_type]
        elif box.attributes.bus_type:
            label = CADCObjectClass[box.attributes.bus_type]
        elif box.attributes.bicycle_tye:
            label = CADCObjectClass[box.attributes.bicycle_type]
        else:
            label = CADCObjectClass[box.label]
        if box.attributes.state:
            label = label | CADCObjectClass[box.attributes.state]

        obj_arr.append(ObjectTarget3D(
            [box.position.x, box.position.y, box.position.z],
            Rotation.from_euler('z', box.yaw),
            [box.dimensions.x, box.dimensions.y, box.dimensions.z],
            ObjectTag(label, CADCObjectClass),
            tid=int(box.uuid.replace('-', ''), 16) % (1 << 63)
        ))
    return obj_arr

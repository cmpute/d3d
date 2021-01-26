import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from enum import IntFlag
from pathlib import Path

import numpy as np
from addict import Dict as edict

class Kitti360Class(IntFlag): # XXX: actually is CityscapeClass
    """
    Categories and attributes of an annotation in KITTI-360 dataset.

    The ids are encoded into 2bytes integer::

        0xFF
          │└: category
          └─: label
    """
       
    void                    = 0x00
    unlabeled               = 0x10
    ego_vehicle             = 0x20
    rectification_border    = 0x30
    out_of_roi              = 0x40
    static                  = 0x50
    dynamic                 = 0x60
    ground                  = 0x70
    unknown_construction    = 0x80
    unknown_vehicle         = 0x90
    unknown_object          = 0xa0

    flat                    = 0x01
    road                    = 0x11
    sidewalk                = 0x21
    parking                 = 0x31
    rail_track              = 0x41

    construction            = 0x02
    building                = 0x12
    wall                    = 0x22
    fence                   = 0x32
    guard_rail              = 0x42
    bridge                  = 0x52
    tunnel                  = 0x62
    garage                  = 0x70
    gate                    = 0x80
    stop                    = 0x90

    object_                 = 0x03
    pole                    = 0x13
    polegroup               = 0x23
    traffic_light           = 0x33
    traffic_sign            = 0x43
    smallpole               = 0x50
    lamp                    = 0x60
    trash_bin               = 0x70
    vending_machine         = 0x80
    box                     = 0x90

    nature                  = 0x04
    vegetation              = 0x14
    terrain                 = 0x24

    sky                     = 0x05

    human                   = 0x06
    person                  = 0x16
    rider                   = 0x26

    vehicle                 = 0x07
    car                     = 0x17
    truck                   = 0x27
    bus                     = 0x37
    caravan                 = 0x47
    trailer                 = 0x57
    train                   = 0x67
    motorcycle              = 0x77
    bicycle                 = 0x87
    license_plate           = 0x97


# Following definitions are from https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py

_Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'kittiId'     , # An integer ID that is associated with this label for KITTI-360
                    # NOT FOR RELEASING

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

_labels = [
    #        name                                  id    kittiId,    trainId    category             catId     hasInstances   ignoreInEval   color
    _Label(  Kitti360Class.unlabeled             ,  0 ,       -1 ,       255 ,  "void"             , 0       , False        , True         , (  0,  0,  0) ),
    _Label(  Kitti360Class.ego_vehicle           ,  1 ,       -1 ,       255 ,  "void"             , 0       , False        , True         , (  0,  0,  0) ),
    _Label(  Kitti360Class.rectification_border  ,  2 ,       -1 ,       255 ,  "void"             , 0       , False        , True         , (  0,  0,  0) ),
    _Label(  Kitti360Class.out_of_roi            ,  3 ,       -1 ,       255 ,  "void"             , 0       , False        , True         , (  0,  0,  0) ),
    _Label(  Kitti360Class.static                ,  4 ,       -1 ,       255 ,  "void"             , 0       , False        , True         , (  0,  0,  0) ),
    _Label(  Kitti360Class.dynamic               ,  5 ,       -1 ,       255 ,  "void"             , 0       , False        , True         , (111, 74,  0) ),
    _Label(  Kitti360Class.ground                ,  6 ,       -1 ,       255 ,  "void"             , 0       , False        , True         , ( 81,  0, 81) ),
    _Label(  Kitti360Class.road                  ,  7 ,        1 ,         0 ,  "flat"             , 1       , False        , False        , (128, 64,128) ),
    _Label(  Kitti360Class.sidewalk              ,  8 ,        3 ,         1 ,  "flat"             , 1       , False        , False        , (244, 35,232) ),
    _Label(  Kitti360Class.parking               ,  9 ,        2 ,       255 ,  "flat"             , 1       , False        , True         , (250,170,160) ),
    _Label(  Kitti360Class.rail_track            , 10 ,        10,       255 ,  "flat"             , 1       , False        , True         , (230,150,140) ),
    _Label(  Kitti360Class.building              , 11 ,        11,         2 ,  "construction"     , 2       , True         , False        , ( 70, 70, 70) ),
    _Label(  Kitti360Class.wall                  , 12 ,        7 ,         3 ,  "construction"     , 2       , False        , False        , (102,102,156) ),
    _Label(  Kitti360Class.fence                 , 13 ,        8 ,         4 ,  "construction"     , 2       , False        , False        , (190,153,153) ),
    _Label(  Kitti360Class.guard_rail            , 14 ,        30,       255 ,  "construction"     , 2       , False        , True         , (180,165,180) ),
    _Label(  Kitti360Class.bridge                , 15 ,        31,       255 ,  "construction"     , 2       , False        , True         , (150,100,100) ),
    _Label(  Kitti360Class.tunnel                , 16 ,        32,       255 ,  "construction"     , 2       , False        , True         , (150,120, 90) ),
    _Label(  Kitti360Class.pole                  , 17 ,        21,         5 ,  "object"           , 3       , True         , False        , (153,153,153) ),
    _Label(  Kitti360Class.polegroup             , 18 ,       -1 ,       255 ,  "object"           , 3       , False        , True         , (153,153,153) ),
    _Label(  Kitti360Class.traffic_light         , 19 ,        23,         6 ,  "object"           , 3       , True         , False        , (250,170, 30) ),
    _Label(  Kitti360Class.traffic_sign          , 20 ,        24,         7 ,  "object"           , 3       , True         , False        , (220,220,  0) ),
    _Label(  Kitti360Class.vegetation            , 21 ,        5 ,         8 ,  "nature"           , 4       , False        , False        , (107,142, 35) ),
    _Label(  Kitti360Class.terrain               , 22 ,        4 ,         9 ,  "nature"           , 4       , False        , False        , (152,251,152) ),
    _Label(  Kitti360Class.sky                   , 23 ,        9 ,        10 ,  "sky"              , 5       , False        , False        , ( 70,130,180) ),
    _Label(  Kitti360Class.person                , 24 ,        19,        11 ,  "human"            , 6       , True         , False        , (220, 20, 60) ),
    _Label(  Kitti360Class.rider                 , 25 ,        20,        12 ,  "human"            , 6       , True         , False        , (255,  0,  0) ),
    _Label(  Kitti360Class.car                   , 26 ,        13,        13 ,  "vehicle"          , 7       , True         , False        , (  0,  0,142) ),
    _Label(  Kitti360Class.truck                 , 27 ,        14,        14 ,  "vehicle"          , 7       , True         , False        , (  0,  0, 70) ),
    _Label(  Kitti360Class.bus                   , 28 ,        34,        15 ,  "vehicle"          , 7       , True         , False        , (  0, 60,100) ),
    _Label(  Kitti360Class.caravan               , 29 ,        16,       255 ,  "vehicle"          , 7       , True         , True         , (  0,  0, 90) ),
    _Label(  Kitti360Class.trailer               , 30 ,        15,       255 ,  "vehicle"          , 7       , True         , True         , (  0,  0,110) ),
    _Label(  Kitti360Class.train                 , 31 ,        33,        16 ,  "vehicle"          , 7       , True         , False        , (  0, 80,100) ),
    _Label(  Kitti360Class.motorcycle            , 32 ,        17,        17 ,  "vehicle"          , 7       , True         , False        , (  0,  0,230) ),
    _Label(  Kitti360Class.bicycle               , 33 ,        18,        18 ,  "vehicle"          , 7       , True         , False        , (119, 11, 32) ),
    _Label(  Kitti360Class.garage                , 34 ,        12,         2 ,  "construction"     , 2       , True         , False        , ( 64,128,128) ),
    _Label(  Kitti360Class.gate                  , 35 ,        6 ,         4 ,  "construction"     , 2       , False        , False        , (190,153,153) ),
    _Label(  Kitti360Class.stop                  , 36 ,        29,       255 ,  "construction"     , 2       , True         , True         , (150,120, 90) ),
    _Label(  Kitti360Class.smallpole             , 37 ,        22,         5 ,  "object"           , 3       , True         , False        , (153,153,153) ),
    _Label(  Kitti360Class.lamp                  , 38 ,        25,       255 ,  "object"           , 3       , True         , False        , (0,   64, 64) ),
    _Label(  Kitti360Class.trash_bin             , 39 ,        26,       255 ,  "object"           , 3       , True         , False        , (0,  128,192) ),
    _Label(  Kitti360Class.vending_machine       , 40 ,        27,       255 ,  "object"           , 3       , True         , False        , (128, 64,  0) ),
    _Label(  Kitti360Class.box                   , 41 ,        28,       255 ,  "object"           , 3       , True         , False        , (64,  64,128) ),
    _Label(  Kitti360Class.unknown_construction  , 42 ,        35,       255 ,  "void"             , 0       , False        , True         , (102,  0,  0) ),
    _Label(  Kitti360Class.unknown_vehicle       , 43 ,        36,       255 ,  "void"             , 0       , False        , True         , ( 51,  0, 51) ),
    _Label(  Kitti360Class.unknown_object        , 44 ,        37,       255 ,  "void"             , 0       , False        , True         , ( 32, 32, 32) ),
    _Label(  Kitti360Class.license_plate         , -1 ,        -1,        -1 ,  "vehicle"          , 7       , False        , True         , (  0,  0,142) ),
]

kittiId2label = { label.kittiId : label for label in _labels }
id2label =      { label.id      : label for label in _labels }


def load_sick_scan(basepath, file):
    if isinstance(basepath, (str, Path)):
        scan = np.fromfile(Path(basepath, file), dtype=np.float32)
    else:
        with basepath.open(str(file)) as fin:
            buffer = fin.read()
        scan = np.frombuffer(buffer, dtype=np.float32)
    return scan.reshape((-1, 2))


def load_bboxes(basepath, file):
    """
    :param poses: if poses are provided, static objects will be filtered if it's out
        of visible_range
    """    
    from intervaltree import Interval, IntervalTree

    if isinstance(basepath, (str, Path)):
        tree = ET.parse(Path(basepath, file))
        root = tree.getroot()
    else:  # assume ZipFile object
        root = ET.fromstring(basepath.read(str(file)))

    frame_mapping = []
    object_list = []
    for idx, child in enumerate(root):
        if not child.tag.startswith("object"):
            continue

        obj = edict()
        for prop in child:
            if prop.tag == "transform":
                obj.transform = np.fromstring(prop.find("data").text,
                                              dtype=float, sep=" ").reshape(4, 4)
            elif prop.tag == "vertices":
                obj.vertices = np.fromstring(prop.find("data").text,
                                             dtype=float, sep=" ").reshape(-1, 3)
            elif prop.tag == "faces":
                obj.faces = np.fromstring(prop.find("data").text,
                                          dtype=int, sep=" ").reshape(-1, 3)
            elif prop.tag not in ["label", "category"]:
                obj[prop.tag] = int(prop.text)
            else:
                obj[prop.tag] = prop.text

        object_list.append(obj)
        if obj.dynamic:
            frame_mapping.append(Interval(obj.timestamp, obj.timestamp+1, idx))
        else:
            frame_mapping.append(Interval(obj.start_frame, obj.end_frame, idx))

    return object_list, IntervalTree(frame_mapping)

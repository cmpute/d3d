import os
import os.path as osp
from enum import Enum
from zipfile import ZipFile

import numpy as np

from scipy.spatial.transform import Rotation
from d3d.dataset.kitti import utils
from d3d.abstraction import ObjectTarget3D, ObjectTag, ObjectTarget3DArray

class KittiObjectClass(Enum):
    DontCare = 0
    Car = 1
    Van = 2
    Truck = 3
    Pedestrian = 4
    Person_sitting = 5
    Cyclist = 6
    Tram = 7
    Misc = 255

class ObjectLoader:
    """
    Load and parse odometry benchmark data into a usable format.
    Please ensure that calibration and poses are downloaded
    
    # Zip Files
    - data_object_calib.zip
    - data_object_image_2.zip
    - data_object_image_3.zip
    - data_object_label_2.zip
    - data_object_velodyne.zip

    # Unzipped Structure
    - <base_path directory>
        - training
            - calib
            - image_2
            - label_2
            - velodyne
        - testing
            - calib
            - image_2
            - velodyne

    :param phase: training or testing
    """

    def __init__(self, base_path, frames=None, inzip=False, phase="training"):
        self.base_path = base_path
        self.frames = frames
        self.inzip = inzip
        self.phase = phase

        self.cam2_data = None
        self.cam3_data = None
        self.velo_data = None
        self.calib_data = None
        self.label_data = None
        total_count = None

        if phase not in ['training', 'testing']:
            raise ValueError("Invalid phase tag")

        # Open zipfiles and count total number of frames
        if self.inzip:
            if phase == 'training':
                # label data is necessary when training
                self.label_data = ZipFile(osp.join(base_path, "data_object_label_2.zip"))
                total_count = sum(1 for name in self.label_data.namelist() if not name.endswith('/'))
            if osp.exists(osp.join(base_path, "data_object_image_2.zip")):
                self.cam2_data = ZipFile(osp.join(base_path, "data_object_image_2.zip"))
                total_count = total_count or sum(1 for name in self.cam2_data.namelist() if not name.endswith('/'))
            if osp.exists(osp.join(base_path, "data_object_image_3.zip")):
                self.cam3_data = ZipFile(osp.join(base_path, "data_object_image_3.zip"))
            if osp.exists(osp.join(base_path, "data_object_velodyne.zip")):
                self.velo_data = ZipFile(osp.join(base_path, "data_object_velodyne.zip"))
                total_count = total_count or sum(1 for name in self.velo_data.namelist() if not name.endswith('/'))
            if osp.exists(osp.join(base_path, "data_object_calib.zip")):
                self.calib_data = ZipFile(osp.join(base_path, "data_object_calib.zip"))
        else:
            self.data_path = os.path.join(base_path, phase)
            if phase == 'training':
                total_count = len(os.listdir(osp.join(self.data_path, 'label_2')))
            elif os.path.exists(osp.join(self.data_path, 'image_2')):
                total_count = len(os.listdir(osp.join(self.data_path, 'image_2')))
            elif os.path.exists(osp.join(self.data_path, 'velodyne')):
                total_count = len(os.listdir(osp.join(self.data_path, 'velodyne')))

        # Find all the data files
        self.frames = self.frames or list(range(total_count))

    def __len__(self):
        return len(self.frames)

    def close(self):
        if self.inzip:
            if self.label_data is not None:
                self.label_data.close()
            if self.cam2_data is not None:
                self.cam2_data.close()
            if self.cam3_data is not None:
                self.cam3_data.close()
            if self.velo_data is not None:
                self.velo_data.close()
            if self.calib_data is not None:
                self.calib_data.close()

    # ========== Generators ==========

    def cam2(self, idx=None):
        source = self.cam2_data if self.inzip else self.data_path
        if idx is None:
            return utils.yield_images(source, [osp.join('cam_2', '%06d.png' % fid) for fid in self.frames], gray=False)
        else:
            return utils.load_image(source, osp.join('cam_2', '%06d.png' % idx), gray=False)

    def cam3(self, idx=None):
        source = self.cam3_data if self.inzip else self.data_path
        if idx is None:
            return utils.yield_images(source, [osp.join('cam_3', '%06d.png' % fid) for fid in self.frames], gray=False)
        else:
            return utils.load_image(source, osp.join('cam_3', '%06d.png' % idx), gray=False)

    def velo(self, idx=None):
        source = self.velo_data if self.inzip else self.data_path
        if idx is None:
            return utils.yield_velo_scans(source, [osp.join('velodyne', '%06d.bin' % fid) for fid in self.frames])
        else:
            return utils.load_velo_scan(source, osp.join('velodyne', '%06d.bin' % idx))

    def calib(self, idx=None):
        source = self.calib_data if self.inzip else self.data_path
        if idx is None:
            return self._yield_calibs(source, [osp.join('calib', '%06d.txt' % fid) for fid in self.frames])
        else:
            return self._load_calib(source, osp.join('calib', '%06d.txt' % idx))

    def label(self, idx=None):
        if self.phase == "testing":
            raise ValueError("Testing dataset doesn't contain label data")
        source = self.label_data if self.inzip else self.data_path
        if idx is None:
            return self._yield_labels(source, [osp.join('label_2', '%06d.txt' % fid) for fid in self.frames])
        else:
            return self._load_label(source, osp.join('label_2', '%06d.txt' % idx))

    def label_objects(self, idx=None):
        '''
        Return list of converted ground truth targets. Objects labelled as `DontCare` are removed
        '''
        if idx is None:
            return self._yield_objects()
        else:
            return self._generate_objects(self.label(idx), self.calib(idx))

    # ========== Implementations ==========

    def _yield_calibs(self, basepath, filelist):
        for file in filelist:
            yield self._load_calib(basepath, file)

    def _load_calib(self, basepath, file):
        data = {}

        # Load the calibration file
        filedata = utils.load_calib_file(basepath, file)

        # Reshape matrices
        data['P0'] = filedata['P0'].reshape(3, 4)
        data['P1'] = filedata['P1'].reshape(3, 4)
        data['P2'] = filedata['P2'].reshape(3, 4)
        data['P3'] = filedata['P3'].reshape(3, 4)
        data['R0_rect'] = filedata['R0_rect'].reshape(3, 3)
        data['Tr_velo_to_cam'] = filedata['Tr_velo_to_cam'].reshape(3, 4)

        return data

    def _yield_labels(self, basepath, filelist):
        for file in filelist:
            yield self._load_label(basepath, file)

    def _load_label(self, basepath, file):
        data = []
        if isinstance(basepath, str):
            fin = open(os.path.join(basepath, file))
        else: # assume ZipFile object
            fin = basepath.open(file)

        with fin:
            for line in fin.readlines():
                if not line.strip():
                    continue

                values = [KittiObjectClass[value] if idx == 0 else float(value)
                    for idx, value in enumerate(line.split(' '))]
                data.append(values)

        return data

    def _yield_objects(self):
        for label, calib in zip(self.label(), self,calib()):
            yield self._generate_objects(label, calib)

    def _generate_objects(self, label, calib):
        Tr = calib['Tr_velo_to_cam']
        RRect = Rotation.from_dcm(calib['R0_rect'])
        HR, HT = Rotation.from_dcm(Tr[:,:3]), Tr[:,3]
        objects = ObjectTarget3DArray()

        for label in label:
            if label[0] == KittiObjectClass.DontCare:
                continue

            h,w,l = label[8:11] # height, width, length
            position = label[11:14] # x, y, z in camera coordinate
            ry = label[14] # rotation of y axis in camera coordinate
            position[1] -= h/2

            position = np.dot(position, RRect.inv().as_dcm().T)
            position = HR.inv().as_dcm().dot(position - HT)
            orientation = HR.inv() * RRect.inv() * Rotation.from_euler('y', ry)
            orientation *= Rotation.from_euler("x", np.pi/2) # change dimension from l,h,w to l,w,h

            tag = ObjectTag(label[0], KittiObjectClass)
            target = ObjectTarget3D(position, orientation, [l,w,h], tag)
            objects.targets.append(target)

        return objects

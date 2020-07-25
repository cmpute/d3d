import os
import random
import unittest

import numpy as np
import pcl
from matplotlib import pyplot as plt
from tkinter import TclError

from d3d.dataset.kitti import (KittiObjectClass, KittiObjectLoader,
                                      dump_detection_output, KittiTrackingLoader)
from d3d.dataset.waymo.loader import WaymoObjectLoader
from d3d.dataset.nuscenes.loader import NuscenesObjectClass, NuscenesObjectLoader, NuscenesDetectionClass
from d3d.vis.pcl import visualize_detections as pcl_vis
from d3d.vis.image import visualize_detections as img_vis

# set the location of the dataset in environment variable
# if not set, then the corresponding test will be skipped
# TODO: create mini samples for these datasets to test
kitti_location = os.environ['KITTI'] if 'KITTI' in os.environ else None
waymo_location = os.environ['WAYMO'] if 'WAYMO' in os.environ else None
nuscenes_location = os.environ['NUSCENES'] if 'NUSCENES' in os.environ else None
selection = int(os.environ['INDEX']) if 'INDEX' in os.environ else None

if 'INZIP' in os.environ:
    if os.environ['INZIP'].lower() in ['0', 'false']:
        inzip = False
    elif os.environ['INZIP'].lower() in ['1', 'true']:
        inzip = True
    else:
        raise ValueError("Invalid INZIP option!")
else:
    inzip = True

class CommonObjectDSMixin:
    def test_point_cloud_projection(self):
        idx = selection or random.randint(0, len(self.loader))
        cam = random.choice(self.loader.VALID_CAM_NAMES)
        lidar = random.choice(self.loader.VALID_LIDAR_NAMES)

        cloud = self.loader.lidar_data(idx, lidar)
        image = self.loader.camera_data(idx, cam)
        calib = self.loader.calibration_data(idx)

        uv, mask = calib.project_points_to_camera(cloud, cam, lidar)
        plt.figure(num="Please check whether the lidar points are aligned")
        plt.imshow(image)
        plt.scatter(uv[:,0], uv[:,1], s=2, c=cloud[mask, 3])
        plt.draw()
        try:
            plt.pause(5)
        except TclError: # skip error if manually closed
            pass

    def test_ground_truth_visualizer_pcl(self):
        idx = selection or random.randint(0, len(self.loader))
        lidar = random.choice(self.loader.VALID_LIDAR_NAMES)

        cloud = self.loader.lidar_data(idx, lidar)
        cloud = pcl.create_xyzi(cloud[:, :4])
        targets = self.loader.lidar_objects(idx)
        calib = self.loader.calibration_data(idx)

        visualizer = pcl.Visualizer()
        visualizer.addPointCloud(cloud, field="intensity")
        pcl_vis(visualizer, lidar, targets, calib)
        visualizer.setRepresentationToWireframeForAllActors()
        visualizer.setWindowName("Please check whether the gt boxes are aligned!")
        visualizer.spinOnce(time=5000)
        visualizer.close()

    def test_ground_truth_visualizer_img(self):
        idx = selection or random.randint(0, len(self.loader))
        cam = random.choice(self.loader.VALID_CAM_NAMES)

        image = np.array(self.loader.camera_data(idx, cam))
        targets = self.loader.lidar_objects(idx)
        calib = self.loader.calibration_data(idx)
        
        fig, ax = plt.subplots(num="Please check whether the bounding boxes are aligned")
        plt.imshow(image)
        img_vis(ax, cam, targets, calib, box_color=(1, 1, 0, 0.5))
        plt.draw()
        try:
            plt.pause(5)
        except TclError: # skip error if manually closed
            pass

@unittest.skipIf(not kitti_location, "Path to kitti not set")
class TestKittiObjectDataset(unittest.TestCase, CommonObjectDSMixin):
    def setUp(self):
        self.loader = KittiObjectLoader(kitti_location, inzip=inzip)

    def test_detection_output(self):
        idx = selection or random.randint(0, len(self.loader))
        print("index: ", idx) # for debug
        targets = self.loader.lidar_objects(idx)
        label = self.loader.lidar_objects(idx, raw=True)
        output = dump_detection_output(targets,
            self.loader.calibration_data(idx), self.loader.calibration_data(idx, raw=True))

        # These are for debug. Actually there are some labels in KITTI (usually pedestrian)
        #     whose 2D coordinates are not calculated from 3D box...
        # with open("test_out.txt", "w") as fout:
        #     fout.write(output)
        # with open("test_out_gt.txt", "w") as fout:
        #     fout.write("\n".join([" ".join(map(str, r)) for r in label]))

        output_list = []
        for line in output.split("\n"):
            line = line.split(" ")
            line[0] = KittiObjectClass[line[0]]
            line[1:] = [float(v) for v in line[1:]]
            output_list.append(line)

        for i, oline in enumerate(output_list):
            for j, v in enumerate(oline):
                if j in [1,2,3,15]:
                    continue

                if isinstance(v, KittiObjectClass):
                    assert v == label[i][j]
                elif label[i][j] != 0:
                    diff = abs(v - label[i][j]) / label[i][j]
                    assert diff < 0.05, "[@{}] {} != {}".format(j, v, label[i][j])
                else:
                    assert v == 0

@unittest.skipIf(not waymo_location, "Path to waymo not set")
class TestWaymoObjectDataset(unittest.TestCase, CommonObjectDSMixin):
    def setUp(self):
        self.loader = WaymoObjectLoader(waymo_location, inzip=inzip)

    def test_point_cloud_projection_all(self):
        idx = selection or random.randint(0, len(self.loader))
        cam = random.choice(self.loader.VALID_CAM_NAMES)

        cloud = self.loader.lidar_data(idx, concat=True)
        image = self.loader.camera_data(idx, cam)
        calib = self.loader.calibration_data(idx)

        uv, mask = calib.project_points_to_camera(cloud, cam)
        plt.figure(num="Please check whether the lidar points are aligned")
        plt.imshow(image)
        plt.scatter(uv[:,0], uv[:,1], s=2, c=np.tanh(cloud[mask, 3]))
        plt.draw()
        try:
            plt.pause(5)
        except TclError: # skip error if manually closed
            pass

    def test_ground_truth_visualizer_pcl(self):
        # this function is overrided since point cloud return from waymo is in vehicle frame
        idx = selection or random.randint(0, len(self.loader))

        cloud = self.loader.lidar_data(idx, concat=True)
        cloud = pcl.create_xyzi(cloud[:, :4])
        targets = self.loader.lidar_objects(idx)
        calib = self.loader.calibration_data(idx)

        visualizer = pcl.Visualizer()
        visualizer.addPointCloud(cloud, field="intensity")
        pcl_vis(visualizer, "vehicle", targets, calib)
        visualizer.setRepresentationToWireframeForAllActors()
        visualizer.setWindowName("Please check whether the gt boxes are aligned!")
        visualizer.spinOnce(time=5000)
        visualizer.close()

@unittest.skipIf(not nuscenes_location, "Path to nuscenes not set")
class TestNuscenesObjectDataset(unittest.TestCase, CommonObjectDSMixin):
    def setUp(self):
        self.loader = NuscenesObjectLoader(nuscenes_location, inzip=inzip)
    
    def test_class_parsing(self):
        # test class conversion consistency
        categories = ["animal", "human.pedestrian.adult", "human.pedestrian.child",
            "human.pedestrian.construction_worker", "human.pedestrian.personal_mobility",
            "human.pedestrian.police_officer", "human.pedestrian.stroller",
            "human.pedestrian.wheelchair", "movable_object.barrier",
            "movable_object.debris", "movable_object.pushable_pullable",
            "movable_object.trafficcone", "vehicle.bicycle",
            "vehicle.bus.bendy", "vehicle.bus.rigid",
            "vehicle.car", "vehicle.construction",
            "vehicle.emergency.ambulance", "vehicle.emergency.police",
            "vehicle.motorcycle", "vehicle.trailer",
            "vehicle.truck", "static_object.bicycle_rack"]
        attributes =["vehicle.moving", "vehicle.stopped",
            "vehicle.parked", "cycle.with_rider",
            "cycle.without_rider", "pedestrian.sitting_lying_down",
            "pedestrian.standing", "pedestrian.moving"]

        for name in categories:
            assert NuscenesObjectClass.parse(name).category_name == name
        for name in attributes:
            assert NuscenesObjectClass.parse(name).attribute_name == name

        # test conversion to detection class
        assert NuscenesObjectClass.vehicle_bus_bendy.to_detection() == NuscenesDetectionClass.bus
        assert NuscenesObjectClass.movable_object_trafficcone.to_detection() == NuscenesDetectionClass.traffic_cone
        assert NuscenesObjectClass.animal.to_detection() == NuscenesDetectionClass.ignore

@unittest.skipIf(not kitti_location, "Path to kitti not set")
class TestKittiTrackingDataset(unittest.TestCase):
    def setUp(self):
        self.loader = KittiTrackingLoader(kitti_location, inzip=inzip, nframes=1)

    def test_point_cloud_projection(self):
        idx = selection or random.randint(0, len(self.loader))
        cam = random.choice(self.loader.VALID_CAM_NAMES)
        lidar = random.choice(self.loader.VALID_LIDAR_NAMES)

        cloud1, cloud2 = self.loader.lidar_data(idx, lidar)
        image1, image2 = self.loader.camera_data(idx, cam)
        calib = self.loader.calibration_data(idx)

        fig, (ax1, ax2) = plt.subplots(2, 1, num="Please check whether the lidar points are aligned")
        uv1, mask1 = calib.project_points_to_camera(cloud1, cam, lidar)
        ax1.imshow(image1)
        ax1.scatter(uv1[:,0], uv1[:,1], s=2, c=cloud1[mask1, 3])
        ax1.set_xlim([0, 1242])
        ax1.set_ylim([375, 0])
        
        uv2, mask2 = calib.project_points_to_camera(cloud2, cam, lidar)
        ax2.imshow(image2)
        ax2.scatter(uv2[:,0], uv2[:,1], s=2, c=cloud2[mask2, 3])
        ax2.set_xlim([0, 1242])
        ax2.set_ylim([375, 0])
        
        fig.canvas.draw_idle()
        try TclError:
            plt.pause(5)
        except: # skip error if manually closed
            pass

    def test_ground_truth_visualizer_pcl(self):
        idx = selection or random.randint(0, len(self.loader))
        lidar = random.choice(self.loader.VALID_LIDAR_NAMES)

        # load data
        cloud1, cloud2 = self.loader.lidar_data(idx, lidar)
        pose1, pose2 = self.loader.pose(idx)
        targets1, targets2 = self.loader.lidar_objects(idx)
        calib = self.loader.calibration_data(idx)

        # transform the second frame
        # TODO: haven't tested for very large offset
        trans = pose2.position - pose1.position
        rot = pose2.orientation * pose1.orientation.inv()
        rot_mt = rot.as_matrix().T
        cloud = np.concatenate([np.dot(cloud1[:,:3] + trans, rot_mt), cloud2[:,:3]])
        cloud = np.concatenate([cloud, np.concatenate([cloud1[:,[3]], cloud2[:,[3]]])], axis=1)
        cloud = pcl.create_xyzi(cloud)
        for target in targets1:
            target.position = np.dot(target.position + trans, rot_mt)
            target.orientation = rot * target.orientation

        visualizer = pcl.Visualizer()
        visualizer.addPointCloud(cloud, field="intensity")
        pcl_vis(visualizer, lidar, targets1, calib)
        pcl_vis(visualizer, lidar, targets2, calib, id_prefix="frame2")
        visualizer.setRepresentationToWireframeForAllActors()
        visualizer.setWindowName("Please check whether the gt boxes are aligned!")
        visualizer.spinOnce(time=5000)
        visualizer.close()

if __name__ == "__main__":
    TestKittiObjectDataset().test_detection_output()

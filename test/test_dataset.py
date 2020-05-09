import os
import random
import unittest

import pcl
from matplotlib import pyplot as plt
import time

from d3d.dataset.kitti.object import (KittiObjectClass, KittiObjectLoader,
                                      dump_detection_output)
from d3d.dataset.waymo.loader import WaymoObjectLoader
from d3d.dataset.nuscenes.loader import NuscenesObjectClass, NuscenesObjectLoader, NuscenesDetectionClass
from d3d.vis.pcl import visualize_detections

# set the location of the dataset in environment variable
# if not set, then the corresponding test will be skipped
kitti_location = os.environ['KITTI'] if 'KITTI' in os.environ else None
waymo_location = os.environ['WAYMO'] if 'WAYMO' in os.environ else None
nuscenes_location = os.environ['NUSCENES'] if 'NUSCENES' in os.environ else None
inzip = os.environ['INZIP'] if 'INZIP' in os.environ else True

class CommonMixin:
    def test_point_cloud_projection(self):
        idx = random.randint(0, len(self.loader))
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
        except: # skip error if manually closed
            pass

    def test_ground_truth_visualizer(self):
        idx = random.randint(0, len(self.loader))
        lidar = random.choice(self.loader.VALID_LIDAR_NAMES)

        cloud = self.loader.lidar_data(idx, lidar)
        cloud = pcl.create_xyzi(cloud[:, :4])
        targets = self.loader.lidar_objects(idx)
        calib = self.loader.calibration_data(idx)

        visualizer = pcl.Visualizer()
        visualizer.addPointCloud(cloud, field="intensity")
        visualize_detections(visualizer, lidar, targets, calib)
        visualizer.setRepresentationToWireframeForAllActors()
        visualizer.setWindowName("Please check whether the gt boxes are aligned!")
        visualizer.spinOnce(time=5000)
        visualizer.close()

@unittest.skipIf(not kitti_location, "Path to kitti not set")
class TestKittiDataset(unittest.TestCase, CommonMixin):
    def setUp(self):
        self.loader = KittiObjectLoader(kitti_location, inzip=inzip)

    def test_detection_output(self):
        idx = random.randint(0, len(self.loader))
        targets = self.loader.lidar_objects(idx)
        label = self.loader.lidar_label(idx)
        output = dump_detection_output(targets,
            self.loader.calibration_data(idx), self.loader.calibration_data(idx, raw=True))

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
class TestWaymoDataset(unittest.TestCase, CommonMixin):
    def setUp(self):
        self.loader = WaymoObjectLoader(waymo_location, inzip=inzip)

    def test_ground_truth_visualizer(self):
        # this function is overrided since point cloud return from waymo is in vehicle frame
        idx = random.randint(0, len(self.loader))

        cloud = self.loader.lidar_data(idx)
        cloud = pcl.create_xyzi(cloud[:, :4])
        targets = self.loader.lidar_objects(idx)
        calib = self.loader.calibration_data(idx)

        visualizer = pcl.Visualizer()
        visualizer.addPointCloud(cloud, field="intensity")
        visualize_detections(visualizer, "vehicle", targets, calib)
        visualizer.setRepresentationToWireframeForAllActors()
        visualizer.setWindowName("Please check whether the gt boxes are aligned!")
        visualizer.spinOnce(time=5000)
        visualizer.close()

@unittest.skipIf(not nuscenes_location, "Path to nuscenes not set")
class TestNuscenesDataset(unittest.TestCase, CommonMixin):
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

if __name__ == "__main__":
    TestKittiDataset().test_detection_output()

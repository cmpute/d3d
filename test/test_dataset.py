import os
import random
import unittest
from tkinter import TclError

import numpy as np
import pcl
from d3d.abstraction import EgoPose, Target3DArray, TransformSet
from d3d.dataset.kitti import (KittiObjectClass, KittiObjectLoader,
                               KittiRawLoader, KittiTrackingLoader)
from d3d.dataset.kitti360.loader import KITTI360Loader
from d3d.dataset.nuscenes.loader import (NuscenesDetectionClass,
                                         NuscenesLoader, NuscenesObjectClass)
from d3d.dataset.waymo.loader import WaymoLoader
from d3d.vis.image import visualize_detections as img_vis
from d3d.vis.pcl import visualize_detections as pcl_vis
from matplotlib import pyplot as plt
from PIL.Image import Image

# set the location of the dataset in environment variable
# if not set, then the corresponding test will be skipped
# TODO: create mini samples for these datasets to test
kitti_location = os.environ['KITTI'] if 'KITTI' in os.environ else None
waymo_location = os.environ['WAYMO'] if 'WAYMO' in os.environ else None
nuscenes_location = os.environ['NUSCENES'] if 'NUSCENES' in os.environ else None
kitti360_location = os.environ['KITTI360'] if 'KITTI360' in os.environ else None

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
    # Dataset tester need to define self.oloader as object dataset loader
    def test_accessors(self):
        idx = selection or random.randint(0, len(self.oloader))
        cam = random.choice(self.oloader.VALID_CAM_NAMES)
        lidar = random.choice(self.oloader.VALID_LIDAR_NAMES)

        assert isinstance(self.oloader.lidar_data(idx, lidar), np.ndarray)
        assert isinstance(self.oloader.camera_data(idx, cam), Image)
        assert isinstance(self.oloader.calibration_data(idx), TransformSet)
        assert isinstance(self.oloader.annotation_3dobject(idx), Target3DArray)
        self.oloader.identity(idx)

    def test_point_cloud_projection(self):
        idx = selection or random.randint(0, len(self.oloader))
        cam = random.choice(self.oloader.VALID_CAM_NAMES)
        lidar = random.choice(self.oloader.VALID_LIDAR_NAMES)

        cloud = self.oloader.lidar_data(idx, lidar)
        image = self.oloader.camera_data(idx, cam)
        calib = self.oloader.calibration_data(idx)

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
        idx = selection or random.randint(0, len(self.oloader))
        lidar = random.choice(self.oloader.VALID_LIDAR_NAMES)

        cloud = self.oloader.lidar_data(idx, lidar)
        cloud = pcl.create_xyzi(cloud[:, :4])
        targets = self.oloader.annotation_3dobject(idx)
        calib = self.oloader.calibration_data(idx)

        visualizer = pcl.Visualizer()
        visualizer.addPointCloud(cloud, field="intensity")
        pcl_vis(visualizer, lidar, targets, calib)
        visualizer.setRepresentationToWireframeForAllActors()
        visualizer.setWindowName("Please check whether the gt boxes are aligned!")
        visualizer.spinOnce(time=5000)
        visualizer.close()

    def test_ground_truth_visualizer_img(self):
        idx = selection or random.randint(0, len(self.oloader))
        cam = random.choice(self.oloader.VALID_CAM_NAMES)

        image = np.array(self.oloader.camera_data(idx, cam))
        targets = self.oloader.annotation_3dobject(idx)
        calib = self.oloader.calibration_data(idx)
        
        fig, ax = plt.subplots(num="Please check whether the bounding boxes are aligned")
        plt.imshow(image)
        img_vis(ax, cam, targets, calib, box_color=(1, 1, 0, 0.5))
        plt.draw()
        try:
            plt.pause(5)
        except TclError: # skip error if manually closed
            pass

class CommonTrackingDSMixin:
    # Dataset tester need to define self.tloader as object dataset loader
    def test_accessors(self):
        idx = selection or random.randint(0, len(self.tloader))
        cam = random.choice(self.tloader.VALID_CAM_NAMES)
        lidar = random.choice(self.tloader.VALID_LIDAR_NAMES)

        assert isinstance(self.tloader.lidar_data(idx, lidar)[self.tloader.nframes], np.ndarray)
        assert isinstance(self.tloader.camera_data(idx, cam)[self.tloader.nframes], Image)
        assert isinstance(self.tloader.calibration_data(idx), TransformSet)
        assert isinstance(self.tloader.annotation_3dobject(idx)[self.tloader.nframes], Target3DArray)
        assert isinstance(self.tloader.pose(idx)[self.tloader.nframes], EgoPose)
        assert isinstance(self.tloader.timestamp(idx)[self.tloader.nframes], int)
        self.tloader.identity(idx)

    def test_point_cloud_temporal_fusion(self):
        # TODO: this fail for nuscenes and waymo
        idx = selection or random.randint(0, len(self.tloader))
        lidar = self.tloader.VALID_LIDAR_NAMES[0]

        # load data
        clouds = self.tloader.lidar_data(idx, lidar)
        poses = self.tloader.pose(idx)
        targets = self.tloader.annotation_3dobject(idx)
        calib = self.oloader.calibration_data(idx)

        cloud1, cloud2 = clouds[0][:, :4], clouds[-1][:, :4]
        pose1, pose2 = poses[0], poses[-1]
        targets1, targets2 = targets[0], targets[-1]
        print("START", pose1)
        print("END", pose2)
        
        # create transforms
        tf = TransformSet("global")
        fname1, fname2 = "pose1", "pose2"
        tf.set_intrinsic_map_pin(fname1)
        tf.set_intrinsic_map_pin(fname2)
        tf.set_extrinsic(pose1.homo(), fname1)
        tf.set_extrinsic(pose2.homo(), fname2)

        # make coordinate unified in frame
        targets1 = calib.transform_objects(targets1, lidar)
        targets2 = calib.transform_objects(targets2, lidar)
        targets1.frame = fname1
        targets2.frame = fname2
        print("HOMO1", pose1.homo())
        print("HOMO2", pose2.homo())
        print(tf.get_extrinsic(fname1, fname2))

        # visualize both point cloud in frame2
        visualizer = pcl.Visualizer()
        visualizer.addPointCloud(
            pcl.create_xyzi(tf.transform_points(cloud1, frame_from=fname1, frame_to=fname2)),
            field="intensity", id="cloud1"
        )
        visualizer.addPointCloud(pcl.create_xyzi(cloud2), field="intensity", id="cloud2")
        pcl_vis(visualizer, fname2, targets1, tf, box_color=(1, 1, 0), id_prefix="frame1")
        pcl_vis(visualizer, fname2, targets2, tf, box_color=(0, 1, 1), id_prefix="frame2")
        visualizer.setRepresentationToWireframeForAllActors()
        visualizer.addCoordinateSystem()
        visualizer.setWindowName("Please check whether the gt boxes are aligned!")
        # visualizer.spinOnce(time=5000)
        # visualizer.close()
        visualizer.spin()

@unittest.skipIf(not kitti_location, "Path to kitti not set")
class TestKittiObjectDataset(unittest.TestCase, CommonObjectDSMixin):
    def setUp(self):
        self.oloader = KittiObjectLoader(kitti_location, inzip=inzip)

    def test_detection_output(self):
        idx = selection or random.randint(0, len(self.oloader))
        print("index: ", idx) # for debug
        targets = self.oloader.annotation_3dobject(idx)
        label = self.oloader.annotation_3dobject(idx, raw=True)
        output = self.oloader.dump_detection_output(idx, targets)

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
class TestWaymoDataset(unittest.TestCase, CommonObjectDSMixin, CommonTrackingDSMixin):
    def setUp(self):
        self.oloader = WaymoLoader(waymo_location, inzip=inzip, nframes=0)
        self.tloader = WaymoLoader(waymo_location, inzip=inzip, nframes=3)

    def test_point_cloud_projection_all(self):
        idx = selection or random.randint(0, len(self.oloader))
        cam = random.choice(self.oloader.VALID_CAM_NAMES)

        clouds = self.oloader.lidar_data(idx)
        image = self.oloader.camera_data(idx, cam)
        calib = self.oloader.calibration_data(idx)

        # merge multiple point clouds
        clist = []
        for cloud, frame in zip(clouds, self.oloader.VALID_LIDAR_NAMES):
            clist.append(calib.transform_points(cloud, frame_from=frame, frame_to=None))
        cloud = np.concatenate(clist)

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
        idx = selection or random.randint(0, len(self.oloader))

        cloud = self.oloader.lidar_data(idx, "lidar_top")
        cloud = pcl.create_xyzi(cloud[:, :4])
        targets = self.oloader.annotation_3dobject(idx)
        calib = self.oloader.calibration_data(idx)

        visualizer = pcl.Visualizer()
        visualizer.addPointCloud(cloud, field="intensity")
        pcl_vis(visualizer, "vehicle", targets, calib)
        visualizer.setRepresentationToWireframeForAllActors()
        visualizer.setWindowName("Please check whether the gt boxes are aligned!")
        visualizer.spinOnce(time=5000)
        visualizer.close()


@unittest.skipIf(not nuscenes_location, "Path to nuscenes not set")
class TestNuscenesDataset(unittest.TestCase, CommonObjectDSMixin, CommonTrackingDSMixin):
    def setUp(self):
        self.oloader = NuscenesLoader(nuscenes_location, inzip=inzip, nframes=0)
        self.tloader = NuscenesLoader(nuscenes_location, inzip=inzip, nframes=2)
    
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
class TestKittiTrackingDataset(unittest.TestCase, CommonObjectDSMixin, CommonTrackingDSMixin):
    def setUp(self):
        self.oloader = KittiTrackingLoader(kitti_location, inzip=inzip, nframes=0)
        self.tloader = KittiTrackingLoader(kitti_location, inzip=inzip, nframes=2)


@unittest.skipIf(not kitti_location, "Path to kitti not set")
class TestKittiRawDataset(unittest.TestCase, CommonObjectDSMixin, CommonTrackingDSMixin):
    def setUp(self):
        self.oloader = KittiRawLoader(kitti_location, inzip=inzip, nframes=0)
        self.tloader = KittiRawLoader(kitti_location, inzip=inzip, nframes=2)


@unittest.skipIf(not kitti360_location, "Path to KITTI-360 not set")
class TestKitti360Dataset(unittest.TestCase, CommonObjectDSMixin, CommonTrackingDSMixin):
    def setUp(self):
        self.oloader = KITTI360Loader(kitti360_location, inzip=inzip, nframes=0)
        self.tloader = KITTI360Loader(kitti360_location, inzip=inzip, nframes=2)

    def test_3d_semantic(self):
        seq = "2013_05_28_drive_0000_sync"
        idx = selection or random.randint(0, len(self.oloader))
        cloud = self.oloader.lidar_data((seq, idx), names="velo", bypass=True)
        pose = self.oloader.pose((seq, idx), bypass=True)
        calib = self.oloader.calibration_data((seq, idx))
        cloud = calib.transform_points(cloud[:, :3], frame_to="pose", frame_from="velo")
        cloud = cloud.dot(pose.orientation.as_matrix().T) + pose.position

        labels = np.load(os.path.join(kitti360_location, "data_3d_semantics", seq, "indexed", "%010d.npz" % idx))
        color_cloud = pcl.create_xyzrgb(np.concatenate([cloud[:, :3], labels["rgb"].view('4u1')[:,:3]], axis=1))

        semantic_cloud = pcl.create_xyzl(np.concatenate([cloud[:, :3], labels["semantic"].reshape(-1,1)], axis=1))
        instance_cloud = pcl.create_xyzl(np.concatenate([cloud[:, :3], labels["instance"].reshape(-1,1)], axis=1))
        distance = os.path.join(kitti360_location, "data_3d_semantics", seq, "indexed", "%010d.dist.npy" % idx)
        distance = np.load(distance)
        print("Index:", idx, ", MAX distance", np.max(distance))
        distance_cloud = pcl.create_xyzi(np.concatenate([cloud[:, :3], distance.reshape(-1,1)], axis=1))

        # gt = pcl.io.load_ply("/media/jacob/Storage/Datasets/kitti360/data_3d_semantics/2013_05_28_drive_0000_sync/static/000834_001286.ply")
        # semantic_cloud = pcl.create_xyzl(np.concatenate([gt.xyz, gt.to_ndarray()['semantic'].reshape(-1, 1)], axis=1))
        # instance_cloud = pcl.create_xyzl(np.concatenate([gt.xyz, gt.to_ndarray()['instance'].reshape(-1, 1)], axis=1))

        pcl.io.save_pcd("instance.pcd", instance_cloud, binary=True)
        pcl.io.save_pcd("semantic.pcd", semantic_cloud, binary=True)
        pcl.io.save_pcd("distance.pcd", distance_cloud, binary=True)

        # vis = pcl.Visualizer()
        # vis.addPointCloud(semantic_cloud, field="label")
        # vis.spin()


if __name__ == "__main__":
    TestKittiObjectDataset().test_detection_output()

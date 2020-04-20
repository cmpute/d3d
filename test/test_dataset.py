import unittest

import pcl

from d3d.dataset.kitti.object import KittiObjectLoader, print_detection_result, KittiObjectClass
from d3d.vis.pcl import visualize_detections

kitti_location = "/home/jacobz/PointCloud/detection3/data"

class TestKittiDataset(unittest.TestCase):
    def test_ground_truth_visualizer(self):
        idx = 7
        loader = KittiObjectLoader(kitti_location)
        cloud = loader.velo(idx)
        cloud = pcl.create_xyzi(cloud)
        targets = loader.lidar_objects(idx)

        visualizer = pcl.Visualizer()
        visualizer.addPointCloud(cloud, field="intensity")
        visualize_detections(visualizer, targets)
        visualizer.setRepresentationToWireframeForAllActors()
        visualizer.spinOnce()
        visualizer.close()

    def test_detection_output(self):
        idx = 7
        loader = KittiObjectLoader(kitti_location)

        targets = loader.lidar_objects(idx)
        label = loader.lidar_label(idx)
        output = print_detection_result(targets,
            loader.calibration_data(idx), loader.calibration_data(idx, raw=True))

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
                else:
                    diff = abs(v - label[i][j]) / label[i][j]
                    assert diff < 0.05, "[@{}] {} != {}".format(j, v, label[i][j])


if __name__ == "__main__":
    TestKittiDataset().test_detection_output()

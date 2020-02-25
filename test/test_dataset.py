import pcl
from d3d.dataset.kitti import ObjectLoader
from d3d.vis.pcl import visualize_detections

kitti_location = "/home/jacobz/PointCloud/detection3/data"

def test_kitti():
    idx = 7
    loader = ObjectLoader(kitti_location)
    cloud = loader.velo(idx)
    cloud = pcl.create_xyzi(cloud)
    targets = loader.label_objects(idx)

    visualizer = pcl.Visualizer()
    visualizer.addPointCloud(cloud, field="intensity")
    visualize_detections(visualizer, targets)
    visualizer.setRepresentationToWireframeForAllActors()
    visualizer.spin()
    
test_kitti()

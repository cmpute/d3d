from scipy.spatial.transform import Rotation as R

_pcl_available = False
try:
    import pcl
    _pcl_available = True
except:
    pass

def visualize_detections(visualizer, boxes, scores=None, labels=None):
    if not _pcl_available:
        raise RuntimeError("pcl is not available, please check the installation of package pcl.py")

    # convert coordinate
    for i, box in enumerate(boxes):
        position = box[:3]
        size = box[3:6]
        yaw = box[6]

        orientation = R.from_euler('y', box[6]).as_quat()
        orientation = [orientation[3]] + orientation[:3].tolist()

        visualizer.addCube(position, orientation, size[0], size[1], size[2], id="cube%d" % i)
        visualizer.setShapeRenderingProperties(pv.RenderingProperties.Opacity, 0.5, id="cube%d" % i)

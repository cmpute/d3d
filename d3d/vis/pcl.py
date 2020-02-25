from d3d.abstraction import ObjectTarget3DArray
from scipy.spatial.transform import Rotation as R

_pcl_available = False
try:
    import pcl
    import pcl.visualization as pv
    _pcl_available = True
except:
    pass

def visualize_detections(visualizer: pcl.Visualizer, targets: ObjectTarget3DArray):
    if not _pcl_available:
        raise RuntimeError("pcl is not available, please check the installation of package pcl.py")

    # convert coordinate
    for i, target in enumerate(targets):
        orientation = target.orientation.as_quat()
        orientation = [orientation[3]] + orientation[:3].tolist() # To PCL quaternion
        lx, ly, lz = target.dimension

        visualizer.addCube(target.position, orientation, lx, ly, lz, id="target%d" % i)
        visualizer.setShapeRenderingProperties(pv.RenderingProperties.Opacity, 0.8, id="target%d" % i)

        # draw tag
        disp_text = "#%d: %s" % (i, target.tag_name)
        if target.tag_score < 1:
            disp_text += " (%.2f)" % target.tag_score
        disp_pos = list(target.position)
        disp_pos[2] += lz / 2 # lift the text out of box
        visualizer.addText3D(disp_text, disp_pos, text_scale=0.8, color=(1, 0.8, 1), id="target%d/tag" % i)

        # draw orientation
        direction = target.orientation.as_dcm().dot([1,0,0])
        pstart = target.position
        pend = target.position + direction * lx
        visualizer.addLine(pstart, pend, id="target%d/direction" % i)

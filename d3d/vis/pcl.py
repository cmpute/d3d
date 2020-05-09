from d3d.abstraction import ObjectTarget3DArray
from scipy.spatial.transform import Rotation as R

_pcl_available = False
try:
    import pcl
    import pcl.visualization as pv
    _pcl_available = True
except:
    pass

def visualize_detections(visualizer: pcl.Visualizer, visualizer_frame: str, targets: ObjectTarget3DArray, calib,
    text_scale=0.8, box_color=(1, 1, 1), text_color=(1, 0.8, 1), id_prefix=""):
    '''
    Note: To use this visualizer, targets should be in the same frame as the visualizer frame (lidar frame)
    '''
    if not _pcl_available:
        raise RuntimeError("pcl is not available, please check the installation of package pcl.py")

    if id_prefix != "" and not id_prefix.endswith("/"):
        id_prefix = id_prefix + "/"

    # change frame to the same
    if targets.frame != visualizer_frame:
        targets = calib.transform_objects(targets, frame_to=visualizer_frame)

    for i, target in enumerate(targets):
        # convert coordinate
        orientation = target.orientation.as_quat()
        orientation = [orientation[3]] + orientation[:3].tolist() # To PCL quaternion
        lx, ly, lz = target.dimension

        cube_id = (id_prefix + "target%d") % i
        visualizer.addCube(target.position, orientation, lx, ly, lz, id=cube_id)
        visualizer.setShapeRenderingProperties(pv.RenderingProperties.Opacity, 0.8, id=cube_id)
        visualizer.setShapeRenderingProperties(pv.RenderingProperties.Color, box_color, id=cube_id)

        # draw tag
        text_id = (id_prefix + "target%d/tag") % i
        if target.id:
            disp_text = "%s: %s" % (str(target.id)[:6], target.tag_name)
        else:
            disp_text = "#%d: %s" % (i, target.tag_name)
        if target.tag_score < 1:
            disp_text += " (%.2f)" % target.tag_score
        disp_pos = list(target.position)
        disp_pos[2] += lz / 2 # lift the text out of box
        visualizer.addText3D(disp_text, disp_pos,
            text_scale=text_scale, color=text_color, id=text_id)

        # draw orientation
        arrow_id = (id_prefix + "target%d/direction") % i
        direction = target.orientation.as_matrix().dot([1,0,0])
        pstart = target.position
        pend = target.position + direction * lx
        visualizer.addLine(pstart, pend, id=arrow_id)

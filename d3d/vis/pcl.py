import numpy as np

from d3d.abstraction import Target3DArray, TransformSet, TrackingTarget3D

_pcl_available = False
try:
    import pcl
    import pcl.visualization as pv
    from pcl import Visualizer
    _pcl_available = True
except:
    import typing
    Visualizer = typing.Any

# TODO: support set box_color by id hash
def visualize_detections(visualizer: Visualizer, visualizer_frame: str, targets: Target3DArray, calib: TransformSet,
    text_scale=0.8, box_color=(1, 1, 1), text_color=(1, 0.8, 1), id_prefix="", tags=None, text_offset=None):
    '''
    '''
    if not _pcl_available:
        raise RuntimeError("pcl is not available, please check the installation of package pcl.py")

    if id_prefix != "" and not id_prefix.endswith("/"):
        id_prefix = id_prefix + "/"

    # change frame to the same
    if targets.frame != visualizer_frame:
        targets = calib.transform_objects(targets, frame_to=visualizer_frame)

    for i, target in enumerate(targets.filter_tag(tags)):
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
        if target.tid:
            disp_text = "%s: %s" % (hex(target.tid)[2:8], target.tag_name)
        else:
            disp_text = "#%d: %s" % (i, target.tag_name)
        aux_text = []
        if target.tag_score < 1:
            aux_text.append("%.2f" % target.tag_score)
        position_var = np.power(np.linalg.det(target.position_var), 1/6) # display standard deviation
        if position_var > 0:
            aux_text.append("%.2f" % position_var)
        dimension_var = np.power(np.linalg.det(target.dimension_var), 1/6)
        if dimension_var > 0:
            aux_text.append("%.2f" % dimension_var)
        if target.orientation_var > 0:
            aux_text.append("%.2f" % target.orientation_var)
        if len(aux_text) > 0:
            disp_text += " (" + ", ".join(aux_text) + ")"

        disp_pos = np.copy(target.position)
        disp_pos[2] += lz / 2 # lift the text out of box
        if text_offset is not None: # apply offset
            disp_pos += text_offset
        visualizer.addText3D(disp_text, list(disp_pos),
            text_scale=text_scale, color=text_color, id=text_id)

        # draw orientation
        arrow_id = (id_prefix + "target%d/direction") % i
        direction = target.orientation.as_matrix().dot([1,0,0])
        pstart = target.position
        pend = target.position + direction * lx
        visualizer.addLine(pstart, pend, id=arrow_id)

        # draw velocity
        if isinstance(target, TrackingTarget3D):
            arrow_id = (id_prefix + "target%d/velocity") % i
            pstart = target.position
            pend = target.position + target.velocity
            visualizer.addLine(pstart, pend, color=(0,0,1), id=arrow_id)

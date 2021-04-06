import numpy as np
from matplotlib import cm
from matplotlib.colors import Colormap

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


def visualize_detections(visualizer: Visualizer, visualizer_frame: str, targets: Target3DArray, calib: TransformSet,
    text_scale=0.8, box_color=(1, 1, 1), text_color=(1, 0.8, 1), id_prefix="", tags=None, text_offset=None, viewport=0):
    '''
    Visualize detection targets in PCL Visualizer.

    :param visualizer: The pcl.Visualizer instance used for visualization
    :param visualizer_frame: The frame that visualizer is in
    :param targets: Object array to be visualized
    :param calib: TransformSet object storing calibration information. This is mandatory if the
        targets are in different frames
    :param text_scale: The scale for text tags. Set to 0 or negative if you want to suppress text
        visualization
    :param box_color: Specifying the color of bounding boxes.
        If it's a tuple, then it's assumed that it contains three RGB values in range 0-1.
        If it's a str or matplotlib colormap object, then the color comes from applying colormap to the object id.
    :param text_color: Specifying the color of text tags.
    :param id_prefix: Prefix of actor ids in PCL Visualizer, essential when this function is called multiple times
    :param text_offset: Relative position of text tags with regard to the box center
    :param viewport: Viewport for objects to be added. This is a PCL related feature
    '''
    if not _pcl_available:
        raise RuntimeError("pcl is not available, please check the installation of package pcl.py")

    if id_prefix != "" and not id_prefix.endswith("/"):
        id_prefix = id_prefix + "/"

    # change frame to the same
    if targets.frame != visualizer_frame:
        targets = calib.transform_objects(targets, frame_to=visualizer_frame)

    # convert colormaps
    if isinstance(box_color, str):
        box_color = cm.get_cmap(box_color)
    if isinstance(text_color, str):
        text_color = cm.get_cmap(text_color)

    for i, target in enumerate(targets.filter_tag(tags)):
        tid = target.tid or i

        # convert coordinate
        orientation = target.orientation.as_quat()
        orientation = [orientation[3]] + orientation[:3].tolist() # To PCL quaternion
        lx, ly, lz = target.dimension

        cube_id = (id_prefix + "target%d") % i
        color = box_color(tid % 256) if isinstance(box_color, Colormap) else box_color
        alpha = color[3] if len(color) > 3 else 0.8
        visualizer.addCube(target.position, orientation, lx, ly, lz, id=cube_id, viewport=viewport)
        visualizer.setShapeRenderingProperties(pv.RenderingProperties.Opacity, alpha, id=cube_id)
        visualizer.setShapeRenderingProperties(pv.RenderingProperties.Color, color[:3], id=cube_id)

        # draw tag
        if text_scale >= 0:
            text_id = (id_prefix + "target%d/tag") % i
            if target.tid:
                disp_text = "%s: %s" % (target.tid64, target.tag_top.name)
            else:
                disp_text = "#%d: %s" % (i, target.tag_top.name)
            aux_text = []
            if target.tag_top_score < 1:
                aux_text.append("%.2f" % target.tag_top_score)
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

            color = text_color(tid % 256) if isinstance(text_color, Colormap) else text_color
            visualizer.addText3D(disp_text, list(disp_pos),
                text_scale=text_scale, color=text_color[:3], id=text_id, viewport=viewport)

        # draw orientation
        arrow_id = (id_prefix + "target%d/direction") % i
        dir_x, dir_y, dir_z = np.hsplit(target.orientation.as_matrix(), 3)
        off_x, off_y, off_z = dir_x * lx / 2, dir_y * ly / 2, dir_z * lz / 2
        off_x, off_y, off_z = off_x.flatten(), off_y.flatten(), off_z.flatten()
        pos_bottom = target.position - off_z
        visualizer.addLine(pos_bottom - off_y - off_x, pos_bottom + off_x, id=arrow_id+"_1", viewport=viewport)
        visualizer.addLine(pos_bottom + off_y - off_x, pos_bottom + off_x, id=arrow_id+"_2", viewport=viewport)

        # draw velocity
        if isinstance(target, TrackingTarget3D):
            arrow_id = (id_prefix + "target%d/velocity") % i
            pstart = target.position
            pend = target.position + target.velocity
            visualizer.addLine(pstart, pend, color=(0.5, 0.5, 1), id=arrow_id, viewport=viewport)

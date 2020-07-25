'''
This module contains visualization methods on image
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import axes, lines
from d3d.abstraction import ObjectTarget3DArray, TransformSet

def visualize_detections(ax: axes.Axes, image_frame: str, targets: ObjectTarget3DArray, calib: TransformSet,
    box_color=(0, 1, 0), thickness=2, tags=None):
    '''
    Draw detected object on matplotlib canvas
    '''
    for target in targets.filter_tag(tags):
        # add points for direction indicator
        points = target.corners
        indicator = np.array([ 
                [0, 0, -target.dimension[2]/2],
                [target.dimension[0]/2, 0, -target.dimension[2]/2]
            ]).dot(target.orientation.as_matrix().T)
        points = np.vstack([points, target.position + indicator])

        # project points
        uv, mask, dmask = calib.project_points_to_camera(points, frame_to=image_frame, frame_from=targets.frame,
            remove_outlier=False, return_dmask=True)
        if len(uv[mask]) < 1:
            continue # ignore boxes that is outside the image
        uv = uv.astype(int)

        # draw box
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7),
                 (0, 4), (1, 5), (2, 6), (3, 7),
                 (0, 2), (1, 3), (4, 6), (5, 7)]
        inlier = [i in mask for i in range(len(uv))]
        for i, j in pairs:
            if not inlier[i] and not inlier[j]:
                continue
            if i not in dmask or j not in dmask:
                continue # only calculate for points ahead
            ax.add_artist(lines.Line2D((uv[i,0], uv[j,0]), (uv[i,1], uv[j,1]), c=box_color, lw=thickness))
        # draw direction
        ax.add_artist(lines.Line2D((uv[-2,0], uv[-1,0]), (uv[-2,1], uv[-1,1]), c=box_color, lw=thickness))

def visualize_detections_bev(ax: axes.Axes, visualizer_frame: str, targets: ObjectTarget3DArray, calib: TransformSet,
    box_color=(0, 1, 0), thickness=2, tags=None):
    
    # change frame to the same
    if targets.frame != visualizer_frame:
        targets = calib.transform_objects(targets, frame_to=visualizer_frame)

    for target in targets.filter_tag(tags):
        points = target.corners
        pairs = [(0, 1), (2, 3), (0, 2), (1, 3)]
        for i, j in pairs:
            ax.add_artist(lines.Line2D((points[i,0], points[j,0]), (points[i,1], points[j,1]), c=box_color, lw=thickness))

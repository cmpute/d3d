'''
This module contains visualization methods on image
'''

import numpy as np
import cv2
from d3d.abstraction import ObjectTarget3DArray, TransformSet

def visualize_detections(image, image_frame, targets: ObjectTarget3DArray, calib: TransformSet, color=(0, 255, 0), thickness=2):
    '''
    Draw detected object on image
    '''
    for target in targets:
        # calculate corner points
        points = target.corners
        indicator = np.array([ # add points for direction indicator
                [0, 0, -target.dimension[2]/2],
                [target.dimension[0]/2, 0, -target.dimension[2]/2]
            ]).dot(target.orientation.as_matrix().T)
        points = np.vstack([points, target.position + indicator])

        uv, mask, dmask = calib.project_points_to_camera(points, frame_to=image_frame, frame_from=targets.frame,
            remove_outlier=False, return_dmask=True)
        if len(uv[mask]) < 1: continue # ignore boxes that is outside the image
        uv = uv.astype(int)

        # draw box
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7),
                 (0, 4), (1, 5), (2, 6), (3, 7),
                 (0, 2), (1, 3), (4, 6), (5, 7)]
        inlier = [i in mask for i in range(len(uv))]
        for i, j in pairs:
            if not inlier[i] and not inlier[j]:
                continue
            if i not in dmask or j not in dmask: # only calculate for points ahead
                continue
            cv2.line(image, (uv[i,0], uv[i,1]), (uv[j,0], uv[j,1]), color, thickness)
        # draw direction
        cv2.line(image, (uv[-2,0], uv[-2,1]), (uv[-1,0], uv[-1,1]), color, thickness)
    return image

# TODO: add BEV visualization

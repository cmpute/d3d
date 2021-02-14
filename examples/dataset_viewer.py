
from pathlib import Path
from typing import Union

import numpy as np
from d3d.abstraction import EgoPose
from d3d.dataset.kitti import KittiRawLoader
from d3d.dataset.nuscenes import NuscenesLoader
from d3d.vis.pcl import visualize_detections


def dataset_visualize_pcl(dataset_path: Union[str, Path], dataset_type: str, scene: str, ninter_frames: int = 0):
    '''
    Visualize tracking dataset using PCL visualizer

    :param dataset_path: path to the dataset root
    :param dataset_type: type of supported datasets. Options: {kitti-raw, nuscenes, waymo}
    :param scene: scene ID to be visualized
    '''
    import pcl

    dataset_type = dataset_type.lower()
    if dataset_type == "kitti-raw":
        loader = KittiRawLoader(dataset_path)
    elif dataset_type == "nuscenes":
        loader = NuscenesLoader(dataset_path)
    else:
        raise ValueError("Unsupported dataset type!")
    state = dict(idx=1)

    vis = pcl.Visualizer()
    def render_next(key):
        if not (key is None or (key.KeySym == 'space' and key.keyDown())):
            return

        lidar_frame = loader.VALID_LIDAR_NAMES[0]
        sidx = scene, state['idx']
        objs = loader.annotation_3dobject(sidx)
        calib = loader.calibration_data(sidx)
        cloud = loader.lidar_data(sidx)[:,:4]

        inter_lidar = loader.intermediate_data(sidx, names=lidar_frame, ninter_frames=ninter_frames)
        pose = loader.pose(sidx)
        for frame in inter_lidar:
            lidar_ego_rt = calib.get_extrinsic(frame_from=lidar_frame)
            rt = np.linalg.inv(lidar_ego_rt).dot(np.linalg.inv(pose.homo()))\
                     .dot(frame.pose.homo()).dot(lidar_ego_rt)
            xyz = frame.data[:, :3].dot(rt[:3,:3].T) + rt[:3,3]
            cloud_frame = np.hstack([xyz, frame.data[:,[3]]])
            cloud = np.vstack([cloud, cloud_frame])

        vis.removeAllPointClouds()
        vis.removeAllShapes()
        vis.addPointCloud(pcl.create_xyzi(cloud[:,:4]), field="intensity")
        visualize_detections(vis, lidar_frame, objs, calib, id_prefix="gt", box_color="rainbow")
        vis.setRepresentationToWireframeForAllActors()
        vis.addCoordinateSystem()

        state['idx'] += 1
        if state['idx'] >= loader.sequence_sizes[scene]:
            vis.close()

    vis.registerKeyboardCallback(render_next)
    render_next(None)
    vis.spin()

if __name__ == "__main__":
    import fire
    fire.Fire({
        "pcl": dataset_visualize_pcl
    })

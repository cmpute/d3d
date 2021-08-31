from scipy.spatial.transform.rotation import Rotation
from d3d.dataset.base import MultiModalSequenceDatasetMixin, SequenceDatasetBase
from pathlib import Path
from d3d.abstraction import EgoPose, Target3DArray
from typing import Callable, Union, Optional, List, Any
import tqdm
import numpy as np

try:
    import rospy
    import rosbag
    from geometry_msgs.msg import TransformStamped, Transform
    from sensor_msgs.msg import CameraInfo
    from sensor_msgs.msg import Image as SensorImage
    from tf2_msgs.msg import TFMessage
except ImportError:
    raise ImportError("ROS layout is required for this module!")

import pcl

def dump_sequence_dataset(dataset: SequenceDatasetBase,
                          out_path: Union[str, Path],
                          sequence: Any,
                          size_limit: Optional[int] = None,
                          object_encoder: Callable[[Target3DArray], Any] = None,
                          compression: str = None,
                          odom_frame: str = None,
                          root_name: str = "dataset"):  # TODO: apply root_name before topic
    """
    :param object_encoder: Function to encoder Target3DArray as ROS message. If not present, then it will be serialized as msgpack binary
    :param odom_frame: Which sensor frame to be selected as initial odom pose. If not present, the pose frame will be used for odom
    """
    if isinstance(sequence, list):
        raise ValueError("Only support converting single sequence into ROS bag.")
    if not out_path.endswith(".bag"):
        out_path += ".bag"
    out_path = Path(out_path)

    bag = rosbag.Bag(out_path, "w", compression=compression or "none")

    # write calibration information
    idx0 = sequence, 0
    t0 = rospy.Time.from_sec(dataset.timestamp(idx0) / 1e6)
    tf0 = dataset.pose(idx0)
    calib = dataset.calibration_data(idx0)
    if hasattr(dataset, "VALID_CAM_NAMES"):
        for sensor in dataset.VALID_CAM_NAMES:
            meta = calib.intrinsics_meta[sensor]

            caminfo = CameraInfo()
            caminfo.header.frame_id = sensor
            caminfo.width, caminfo.height = meta.width, meta.height
            caminfo.distortion_model = 'plumb_bob'
            if meta.intri_matrix is not None:
                caminfo.K = meta.intri_matrix.flatten().tolist()
            if meta.distort_coeffs is not None:
                caminfo.D = meta.distort_coeffs.tolist()

            bag.write(f"/camera_data/{sensor}/info", caminfo, t0)

    tfm = TFMessage()
    for name in ([calib.base_frame] + calib.frames):
        if name == dataset.pose_name:
            continue
        tf_msg = TransformStamped()
        tf_msg.header.stamp = t0
        tf_msg.header.frame_id = dataset.pose_name
        tf_msg.child_frame_id = name
        # TransformSet tf is the inverse of ROS tf
        tf = calib.get_extrinsic(frame_to=dataset.pose_name, frame_from=name)
        translation = tf[:3, 3]
        tf_msg.transform.translation.x = translation[0]
        tf_msg.transform.translation.y = translation[1]
        tf_msg.transform.translation.z = translation[2]
        quat = Rotation.from_matrix(tf[:3, :3]).as_quat()
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]
        tfm.transforms.append(tf_msg)

    if odom_frame:
        if odom_frame not in calib.frames and odom_frame != calib.base_frame:
            raise ValueError("Invalid odom frame name!")
        tf_msg = TransformStamped()
        tf_msg.header.stamp = t0
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "odom_pose"
        tf = calib.get_extrinsic(frame_to=odom_frame, frame_from=dataset.pose_name)
        translation = tf[:3, 3]
        tf_msg.transform.translation.x = translation[0]
        tf_msg.transform.translation.y = translation[1]
        tf_msg.transform.translation.z = translation[2]
        quat = Rotation.from_matrix(tf[:3, :3]).as_quat()
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]
        tfm.transforms.append(tf_msg)

    bag.write('/tf_static', tfm, t=t0)

    for i in tqdm.trange(dataset.sequence_sizes[sequence], unit="frames"):
        uidx = sequence, i
        if hasattr(dataset, "VALID_LIDAR_NAMES"):
            for sensor in dataset.VALID_LIDAR_NAMES:
                points = dataset.lidar_data(uidx, names=sensor, formatted=True)
                points_msg = pcl.PointCloud(points).to_msg()
                points_msg.header.frame_id = sensor
                bag.write(f'/lidar_data/{sensor}', points_msg,
                          t=rospy.Time.from_sec(dataset.timestamp(uidx, sensor) / 1e6))
        if hasattr(dataset, "VALID_CAM_NAMES"):
            for sensor in dataset.VALID_CAM_NAMES:
                img = dataset.camera_data(uidx, names=sensor)

                msg = SensorImage()
                msg.height = img.height
                msg.width = img.width

                if img.mode in ['1', 'L']:
                    img = img.convert('L')
                    msg.encoding = "mono8"
                    msg.step = img.width
                else:
                    img = img.convert('RGB')
                    msg.encoding = "rgb8"
                    msg.step = 3 * img.width

                msg.is_bigendian = False
                msg.data = np.array(img).tobytes()
                bag.write(f'/camera_data/{sensor}', msg,
                          t=rospy.Time.from_sec(dataset.timestamp(uidx, sensor) / 1e6))
        
        # TODO: dump objects and semantics

        # write pose
        t_pose = rospy.Time.from_sec(dataset.timestamp(uidx, dataset.pose_name) / 1e6)
        tfm = TFMessage()
        tf_msg = TransformStamped()
        tf_msg.header.stamp = t_pose
        tf_msg.header.frame_id = 'odom_pose' if odom_frame else 'odom'
        tf_msg.child_frame_id = dataset.pose_name

        tf = dataset.pose(uidx)
        tfdiff = np.linalg.inv(tf0.homo()).dot(tf.homo())
        tf_msg.transform.translation.x = tfdiff[0, 3]
        tf_msg.transform.translation.y = tfdiff[1, 3]
        tf_msg.transform.translation.z = tfdiff[2, 3]
        quat = Rotation.from_matrix(tfdiff[:3, :3]).as_quat()
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]

        tfm.transforms.append(tf_msg)
        bag.write('/tf', tfm, t=t_pose)

        if size_limit and out_path.stat().st_size > size_limit:
            print("Terminate because bag size reaches limit.")
            break

    bag.close()
    print("ROS bag creation finished.")

if __name__ == "__main__":
    # from d3d.dataset.cadc import CADCDLoader
    # loader = CADCDLoader("/home/jacobz/Datasets/cadcd-raw", inzip=True)

    from d3d.dataset.kitti import KittiTrackingLoader, KittiOdometryLoader
    loader = KittiOdometryLoader("/media/jacobz/Storage/Datasets/kitti-raw", inzip=True)

    # from d3d.dataset.nuscenes import NuscenesLoader
    # loader = NuscenesLoader("/media/jacobz/Storage/Datasets/nuscenes", inzip=True)

    print(loader.sequence_ids)
    dump_sequence_dataset(loader, "test.bag", loader.sequence_ids[0], size_limit=1e9, odom_frame="velo") #, compression="bz2")
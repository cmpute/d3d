from numpy.core.numeric import load
from numpy.lib.arraysetops import isin
from d3d.dataset.base import MultiModalSequenceDatasetMixin, SequenceDatasetBase
from pathlib import Path
from d3d.abstraction import Target3DArray
from typing import Callable, Union, Optional, List, Any
import tqdm

try:
    import rospy
    import rosbag
except ImportError:
    raise ImportError("rosbag package is required for this module!")

import pcl

def dump_sequence_dataset(dataset: SequenceDatasetBase,
                          out_path: Union[str, Path],
                          sequence: Any,
                          size_limit: Optional[int] = None,
                          object_encoder: Callable[[Target3DArray], Any] = None,
                          root_name: str = "dataset"):
    """
    :param object_encoder: Function to encoder Target3DArray as ROS message. If not present, then it will be serialized as 
    """
    if isinstance(sequence, list):
        raise ValueError("Only support converting single sequence into ROS bag.")
    if not out_path.endswith(".bag"):
        out_path += ".bag"
    bag = rosbag.Bag(out_path, "w")

    for i in tqdm.trange(dataset.sequence_sizes[sequence], unit="frames"):
        if hasattr(dataset, "VALID_LIDAR_NAMES"):
            for sensor in dataset.VALID_LIDAR_NAMES:
                points = dataset.lidar_data(i, names=sensor, formatted=True)
                points_msg = pcl.PointCloud(points).to_msg()
                points_msg.header.frame_id = sensor
                bag.write(f'/lidar_data/{sensor}', points_msg, t=rospy.Time.from_sec(dataset.timestamp(i, sensor) / 1e6))
        if hasattr(dataset, "VALID_CAM_NAMES"):
            # https://github.com/tomas789/kitti2bag/blob/master/kitti2bag/kitti2bag.py#L105
            for sensor in dataset.VALID_CAM_NAMES:
                img = dataset.camera_data(i, names=sensor)

    bag.close()
    print("ROS bag creation finished")

if __name__ == "__main__":
    from d3d.dataset.cadc import CADCDLoader
    loader = CADCDLoader("/home/jacobz/Datasets/cadcd-raw", inzip=True)
    print(loader.sequence_ids)

    dump_sequence_dataset(loader, "test.bag", loader.sequence_ids[0])
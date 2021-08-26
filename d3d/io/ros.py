from d3d.dataset.base import SequenceDatasetBase
from pathlib import Path
from typing import Union, Optional, List

try:
    import rosbag
except ImportError:
    raise ImportError("rosbag package is required for this module!")

import pcl

def dump_sequence_dataset(dataset: SequenceDatasetBase,
                          out_path: Union[str, Path],
                          sequences: Optional[Union[int, List[int]]] = None,
                          size_limit: Optional[int] = None,
                          root_name: str = "dataset"):
    if not out_path.endswith(".bag"):
        out_path += ".bag"
    bag = rosbag.Bag(out_path)

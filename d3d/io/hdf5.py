from d3d.dataset.base import DatasetBase, SequenceDatasetBase
from pathlib import Path
from typing import Union, Optional, List
import tqdm

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required for this module!")

def dump_dataset(dataset: DatasetBase,
                 out_path: Union[str, Path],
                 indices: Optional[Union[int, List[int], slice]] = None,
                 size_limit: Optional[int] = None,
                 root_name: str = "dataset"):
    with h5py.File(out_path,'w') as fhandle:
        root_group = fhandle.create_group(root_name)
        for i in tqdm.trange(len(dataset)):
            seq_group = root_group.create_group("s%d" % i)

            lidar_group = seq_group.create_group("lidar_data")
            points_list = dataset.lidar_data(i, dataset.VALID_LIDAR_NAMES)
            for points, lidar in zip(points_list, dataset.VALID_LIDAR_NAMES):
                lidar_group.create_dataset(lidar, data=points, compression="gzip")

    print("Successfully created dataset")

def dump_sequence_dataset(dataset: SequenceDatasetBase,
                          out_path: Union[str, Path],
                          sequence: Optional[Union[int, List[int]]] = None,
                          size_limit: Optional[int] = None,
                          root_name: str = "dataset"):
    pass

if __name__ == "__main__":
    from d3d.dataset.kitti.tracking import KittiTrackingLoader
    loader = KittiTrackingLoader("/mnt/storage8t/datasets/kitti", inzip=True)
    dump_dataset(loader, "test.h5")

from pathlib import Path
from multiprocessing import Manager, Pool
from typing import Any, List, Optional, Union, Tuple, Dict
from collections import OrderedDict

import numpy as np
import numpy.random as npr
from numpy import ndarray as NdArray
from PIL.Image import Image
from tqdm import tqdm
import functools
import inspect

from d3d.abstraction import Target3DArray, TransformSet, EgoPose

def split_trainval(phase, total_count, trainval_split, trainval_random):
    '''
    split frames for training or validation set
    '''
    if isinstance(trainval_split, list):
        frames = trainval_split
    else:  # trainval_split is a number
        if isinstance(trainval_random, bool):
            frames = npr.default_rng().permutation(total_count) \
                if trainval_random else np.arange(total_count)
        elif isinstance(trainval_random, int):
            gen = npr.default_rng(seed=trainval_random)
            frames = gen.permutation(total_count)
        elif trainval_random == "r":  # predefined shuffle type
            frames = np.arange(total_count)[::-1]
        else:
            raise ValueError("Invalid trainval_random type!")

        if phase == 'training':
            frames = frames[:int(total_count * trainval_split)]
        elif phase == 'validation':
            frames = frames[int(total_count * trainval_split):]

    return frames


def check_frames(names, valid):
    '''
    Check wether names is inside valid options.
    :return: unpack_resule: whether need to unpack results
             names: frame names converted to list
    '''
    unpack_result = False
    if names is None:
        names = valid
    elif isinstance(names, str):
        names = [names]
        unpack_result = True

    # sanity check
    for name in names:
        if name not in valid:
            message = "Invalid frame name %s, " % name
            message += "valid options are " + ", ".join(valid)
            raise ValueError(message)

    return unpack_result, names


class DetectionDatasetBase:
    """
    This class defines basic interface for object detection
    """
    VALID_CAM_NAMES: list
    VALID_LIDAR_NAMES: list

    def __init__(self,
                 base_path: Union[str, Path],
                 inzip: bool = False,
                 phase: str = "training",
                 trainval_split: float = 1.,
                 trainval_random: bool = False):
        """
        :param base_path: directory containing the zip files, or the required data
        :param inzip: whether the dataset is store in original zip archives or unzipped
        :param phase: training or testing
        :param trainval_split: the ratio to split training dataset. If set to 1, then the validation dataset is empty
            If it's a number, then it's the ratio to split training dataset.
            If it's 1, then the validation set is empty, if it's 0, then training set is empty
            If it's a list of number, then it directly defines the indices to report (ignoring trainval_random)
        :param trainval_random: whether select the train/val split randomly.
            If it's a bool, then trainval is split with or without shuffle
            If it's a number, then it's used as the seed for random shuffling
            If it's a string, then predefined order is used. {r: reverse}
        """
        self.base_path = Path(base_path)
        self.inzip = inzip
        self.phase = phase
        # trainval_split and trainval_random should be used only in constructor
        
        if phase not in ['training', 'validation', 'testing']:
            raise ValueError("Invalid phase tag")

    def lidar_data(self,
                   idx: int,
                   names: Optional[Union[str, List[str]]] = None,
                   concat: bool = False
                   ) -> Union[NdArray, List[NdArray]]:
        '''
        Return the lidar point cloud data

        :param names: name of requested lidar frames
        :param idx: index of requested lidar frames
        :param concat: whether to convert the point clouds to base frame and concat them.
                       If only one frame requested, the conversion to base frame will still be performed.
        '''
        raise NotImplementedError("abstract function")

    def camera_data(self,
                    idx: int,
                    names: Optional[Union[str, List[str]]] = None
                    ) -> Union[Image, List[Image]]:
        '''
        Return the camera image data

        :param idx: index of requested image frames
        :param names: name of requested image frames
        '''
        raise NotImplementedError("abstract function")

    def calibration_data(self, idx: int, raw: Optional[bool] = None) -> TransformSet:
        '''
        Return the calibration data

        :param idx: index of requested frame
        :param raw: if false, converted TransformSet will be returned, otherwise raw data will be returned in original format
        '''
        raise NotImplementedError("abstract function")

    def annotation_3dobject(self, idx: int, raw: Optional[bool] = None) -> Target3DArray:
        '''
        Return list of converted ground truth targets in lidar frame.

        :param idx: index of requested frame
        :param raw: if false, targets will be converted to d3d Target3DArray format, otherwise raw data will be returned in original format
        '''
        raise NotImplementedError("abstract function")

    def identity(self, idx: int) -> Any:
        '''
        Return something that can track the data back to original dataset

        :param idx: index of requested frame to be parsed
        '''
        raise NotImplementedError("abstract function")

class TrackingDatasetBase(DetectionDatasetBase):
    """
    Tracking dataset is similarly defined with detection dataset. The two major differences are
    1. Tracking dataset use (sequence_id, frame_id) as identifier.
    2. Tracking dataset provide unique object id across time frames.
    """
    VALID_CAM_NAMES: list
    VALID_LIDAR_NAMES: list

    def __init__(self,
                 base_path: Union[str, Path],
                 inzip: bool = False,
                 phase: str = "training",
                 trainval_split: float = 1.,
                 trainval_random: bool = False,
                 nframes: int = 1):
        """
        :param base_path: directory containing the zip files, or the required data
        :param inzip: whether the dataset is store in original zip archives or unzipped
        :param phase: training or testing
        :param trainval_split: the ratio to split training dataset. If set to 1, then the validation dataset is empty
            If it's a number, then it's the ratio to split training dataset.
            If it's 1, then the validation set is empty, if it's 0, then training set is empty
            If it's a list of number, then it directly defines the indices to report (ignoring trainval_random)
        :param trainval_random: whether select the train/val split randomly.
            If it's a bool, then trainval is split with or without shuffle
            If it's a number, then it's used as the seed for random shuffling
            If it's a string, then predefined order is used. {r: reverse}
        :param nframes: number of consecutive frames returned from the accessors
            If it's a positive number, then it returns adjacent frames with total number reduced
            If it's zero, then it act like object detection dataset, which means the methods will return unpacked data
        """
        super().__init__(base_path, inzip=inzip, phase=phase,
                         trainval_split=trainval_split, trainval_random=trainval_random)
        self.nframes = abs(nframes)

    def _locate_frame(self, idx: int) -> Tuple[int, int]:
        '''
        Subclass should implement this function to convert overall index to (sequence_id, frame_idx) to support
            decorator 'expand_idx' and 'expand_idx_name'
        :return: (seq_id, frame_idx) where frame_idx is the index of starting frame
        '''
        raise NotImplementedError("_locate_frame is not implemented!")

    def lidar_data(self,
                   idx: Union[int, Tuple[int, int]],
                   names: Optional[Union[str, List[str]]] = None,
                   concat: bool = False
                   ) -> Union[NdArray, List[NdArray], List[List[NdArray]]]:
        '''
        If multiple frames are requested, the results will be a list of list. Outer list corresponds to frame names and inner
            list corresponds to time sequence. So names * frames data objects will be returned

        :param names: name of requested lidar frames
        :param idx: index of requested lidar frames
                    if single index is given, then the frame indexing is done on the whole dataset with trainval split
                    if tuple of two integers is given, then first is the sequence index and the second is the frame index,
                    trainval split is ignored in this way and nframes offset is not added
        :param concat: whether to convert the point clouds to base frame and concat them.
                       If only one frame requested, the conversion to base frame will still be performed.
        '''
        raise NotImplementedError("_locate_frame is not implemented!")

    def camera_data(self,
                    idx: Union[int, Tuple[int, int]],
                    names: Optional[Union[str, List[str]]] = None
                    ) -> Union[Image, List[Image], List[List[Image]]]:
        '''
        Return the camera image data

        :param names: name of requested image frames
        :param idx: index of requested lidar frames
        '''
        raise NotImplementedError("abstract function")

    def calibration_data(self, idx: Union[int, Tuple[int, int]], raw: Optional[bool] = False) -> TransformSet:
        '''
        Return the calibration data. Notices that we assume the calibration is fixed among one squence, so it always
            return a single object.

        :param idx: index of requested lidar frames
        :param raw: if false, converted TransformSet will be returned, otherwise raw data will be returned in original format
        '''
        raise NotImplementedError("abstract function")

    def annotation_3dobject(self, idx: Union[int, Tuple[int, int]], raw: Optional[bool] = False) -> Union[Target3DArray, List[Target3DArray]]:
        '''
        Return list of converted ground truth targets in lidar frame.

        :param idx: index of requested frame
        :param raw: if false, targets will be converted to d3d Target3DArray format, otherwise raw data will be returned in original format
        '''
        raise NotImplementedError("abstract function")

    def identity(self, idx: Union[int, Tuple[int, int]]) -> Union[Any, List[Any]]:
        '''
        Return something that can track the data back to original dataset

        :param idx: index of requested frame to be parsed
        :return: if nframes > 0, then the function return a list of ids which are consistent with other functions
        '''
        raise NotImplementedError("abstract function")

    def pose(self, idx: Union[int, Tuple[int, int]], raw: Optional[bool] = False) -> EgoPose:
        '''
        Return (relative) pose of the vehicle for the frame.

        :param idx: index of requested frame
        '''
        raise NotImplementedError("abstract function")

    def timestamp(self, idx: Union[int, Tuple[int, int]], names: Optional[Union[str, List[str]]] = None) -> Union[int, List[int]]:
        '''
        Return the timestamp of frames specified the index, represented by Unix timestamp in miliseconds
        '''
        pass # TODO(v0.4): we should add frame option into this as in KITTI raw dataset, following similar manner

    @property
    def sequence_sizes(self) -> Dict[Any, int]:
        '''
        Return the mapping from sequence id to sequence sizes
        '''
        raise NotImplementedError("abstract function")

    @property
    def sequence_ids(self) -> List[Any]:
        '''
        Return the list of sequence ids
        '''
        raise NotImplementedError("abstract function")


# Some utilities for implementing tracking dataset
# ref: https://stackoverflow.com/questions/2365701/decorating-python-class-methods-how-do-i-pass-the-instance-to-the-decorator


def expand_idx(func):
    '''
    This decorator wraps TrackingDatasetBase member functions with index input. It will delegates the situation
        where self.nframe > 0 to the original function so that the original function can support only one index.
    '''
    @functools.wraps(func)
    def wrapper(self: TrackingDatasetBase, idx, **kwargs):
        if isinstance(idx, int):
            seq_id, frame_idx = self._locate_frame(idx)
        else:
            seq_id, frame_idx = idx

        if self.nframes == 0:
            return func(self, (seq_id, frame_idx), **kwargs)
        else:
            return [func(self, (seq_id, idx), **kwargs)
                    for idx in range(frame_idx, frame_idx + self.nframes + 1)]

    return wrapper

def expand_name(valid_names):
    '''
    This decorator works similar to expand_idx with support to distribute frame names.
    Note that this function acts as a decorator factory instead of decorator
    '''
    def decorator(func):
        default_names = inspect.signature(func).parameters["names"].default
        assert default_names != inspect.Parameter.empty, \
               "The decorated function should have default names value"

        @functools.wraps(func)
        def wrapper(self: TrackingDatasetBase, idx, names=default_names, **kwargs):
            unpack_result, names = check_frames(names, valid_names)

            results = []
            for name in names:
                results.append(func(self, idx, name, **kwargs))
            return results[0] if unpack_result else results

        return wrapper

    return decorator

def expand_idx_name(valid_names):
    '''
    This decorator works similar to expand_idx with support to distribute both indices and frame names.
    Note that this function acts as a decorator factory instead of decorator
    '''
    def decorator(func):
        default_names = inspect.signature(func).parameters["names"].default
        assert default_names != inspect.Parameter.empty, \
               "The decorated function should have default names value"

        @functools.wraps(func)
        def wrapper(self: TrackingDatasetBase, idx, names=default_names, **kwargs):
            if isinstance(idx, int):
                seq_id, frame_idx = self._locate_frame(idx)
            else:
                seq_id, frame_idx = idx
            unpack_result, names = check_frames(names, valid_names)

            results = []
            for name in names:
                if self.nframes == 0:
                    results.append(func(self, (seq_id, frame_idx), name, **kwargs))
                else:
                    results.append([func(self, (seq_id, idx), name, **kwargs)
                                    for idx in range(frame_idx, frame_idx + self.nframes + 1)])
            return results[0] if unpack_result else results

        return wrapper

    return decorator

class NumberPool:
    '''
    This class is a utility for multiprocessing using tqdm
    '''

    def __init__(self, processes, offset=0, *args, **kargs):
        self._ppool = Pool(processes, initializer=tqdm.set_lock,
                           initargs=(tqdm.get_lock(),), *args, **kargs)
        self._npool = Manager().Array('B', [0] * processes)
        self._nlock = Manager().Lock()
        self._offset = offset

    def apply_async(self, func, args=(), callback=None):
        def _wrap_cb(ret):
            n, oret = ret
            with self._nlock:
                self._npool[n] = 0
            if callback is not None:
                callback(oret)

        def _wrap_func(func, args, pool, nlock, offset):
            n = -1
            with nlock:
                n = next(i for i, v in enumerate(pool) if v == 0)
                pool[n] = 1
            ret = func(n + offset, *args)
            return (n, ret)

        self._ppool.apply_async(_wrap_func,
                                (func, args, self._npool,
                                 self._nlock, self._offset),
                                callback=_wrap_cb,
                                error_callback=lambda e: print(
                                    f"{type(e).__name__}: {e}")
                                )

    def close(self):
        self._ppool.close()

    def join(self):
        self._ppool.join()

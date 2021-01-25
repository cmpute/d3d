import functools
import inspect
from collections import OrderedDict
from enum import Enum
from multiprocessing import Manager, Pool
from pathlib import Path
from threading import Event
from typing import (Any, Callable, Dict, List, Optional, OrderedDict, Tuple,
                    Union, ContextManager)

import numpy as np
import numpy.random as npr
from d3d.abstraction import EgoPose, Target3DArray, TransformSet
from numpy import ndarray as NdArray
from PIL.Image import Image
from tqdm import tqdm


def split_trainval(phase: str,
                   total_count: int,
                   trainval_split: Union[float, List[int]],
                   trainval_random: Union[bool, int, str]):
    '''
    Split frames for training or validation set

    :param phase: training or validation
    :param total_count: total number of frames in trainval part of the dataset
    :param trainval_split: the ratio to split training dataset.

        * If it's a number, then it's the ratio to split training dataset.
        * If it's 1, then the validation set is empty; if it's 0, then training set is empty
        * If it's a list of number, then it directly defines the indices to report (ignoring trainval_random)
    :param trainval_random: whether select the train/val split randomly.

        * If it's a bool, then trainval is split with or without shuffle
        * If it's a number, then it's used as the seed for random shuffling
        * If it's a string, then predefined order is used. {r: reverse}
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

        # split phase
        if phase == 'training':
            frames = frames[:int(total_count * trainval_split)]
        elif phase == 'validation':
            frames = frames[int(total_count * trainval_split):]

    return frames

def split_trainval_seq(phase: str,
                       seq_counts: OrderedDict[Any, int],
                       trainval_split: Union[float, List[int]],
                       trainval_random: Union[bool, int, str],
                       by_seq: bool = False):
    '''
    Split frames for training or validation by frames or by sequence

    :param phase: training or validation
    :param total_count: total number of frames in trainval part of the dataset
    :param trainval_split: the ratio to split training dataset.

        * If it's a number, then it's the ratio to split training dataset.
        * If it's 1, then the validation set is empty; if it's 0, then training set is empty
        * If it's a list of number, then it directly defines the indices to report (ignoring :obj:`trainval_random`)
    :param trainval_random: whether select the train/val split randomly.

        * If it's a bool, then trainval is split with or without shuffle
        * If it's a number, then it's used as the seed for random shuffling
        * If it's a string, then predefined order is used. {r: reverse}
    :param by_seq: Whether split trainval partitions by sequences instead of frames
    '''
    if isinstance(trainval_split, list):
        return trainval_split
    if not by_seq:
        total_count = sum(seq_counts.values())
        return split_trainval(phase, total_count, trainval_split, trainval_random)

    seqs = []
    seqstarts = {}
    counter = 0
    for seqid, seqcount in seq_counts.items():
        seqs.append(seqid)
        seqstarts[seqid] = counter
        counter += seqcount

    # generate frames
    frames = []
    if isinstance(trainval_random, bool):
        if trainval_random:
            rng = npr.default_rng()
            seqids = rng.permutation(len(seqs))
            for sid in seqids:
                seq = seqs[sid]
                frames.append(rng.permutation(seq_counts[seq]) + seqstarts[seq])
        else:
            for seq in seqs:
                frames.append(np.arange(seq_counts[seq]) + seqstarts[seq])
    elif isinstance(trainval_random, int):
        rng = npr.default_rng(seed=trainval_random)
        seqids = rng.permutation(len(seqs))
        for sid in seqids:
            seq = seqs[sid]
            frames.append(rng.permutation(seq_counts[seq]) + seqstarts[seq])
    elif trainval_random == "r":  # predefined shuffle type
        for seq in seqs[::-1]:
            frames.append(np.arange(seq_counts[seq])[::-1] + seqstarts[seq])

    # split sequences by phase
    if phase == 'training':
        frames = frames[:int(len(frames) * trainval_split)]
    elif phase == 'validation':
        frames = frames[int(len(frames) * trainval_split):]
    frames = np.concatenate(frames)

    return frames

def check_frames(names: Union[List[str], str], valid: List[str]):
    '''
    Check wether names is inside valid options.

    :param names: Names to be checked
    :param valid: List of the valid names
    :return: unpack_result: whether need to unpack results
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

class DatasetBase:
    """
    This class acts as the base for all dataset loaders

    :param base_path: directory containing the zip files, or the required data
    :param inzip: whether the dataset is store in original zip archives or unzipped
    :param phase: training, validation or testing
    :param trainval_split: the ratio to split training dataset. See
                           documentation of :func:`split_trainval` for detail.
    :param trainval_random: whether select the train/val split randomly. See
                           documentation of :func:`split_trainval` for detail.
    """
    def __init__(self,
                 base_path: Union[str, Path],
                 inzip: bool = False,
                 phase: str = "training",
                 trainval_split: Union[float, List[int]] = 1.,
                 trainval_random: Union[bool, int, str] = False):
        self.base_path = Path(base_path)
        self.inzip = inzip
        self.phase = phase
        # trainval_split and trainval_random should be used only in constructor
        
        if phase not in ['training', 'validation', 'testing']:
            raise ValueError("Invalid phase tag")

        self._return_file_path = False

    class _ReturnPathContext:
        def __init__(self, ds: "DatasetBase"):
            self.ds = ds
        def __enter__(self):
            if self.ds.inzip:
                raise RuntimeError("Cannot return path from a dataset in zip!")
            self.ds._return_file_path = True
        def __exit__(self, type, value, traceback):
            self.ds._return_file_path = False

    def return_path(self) -> ContextManager:
        """
        Make the dataset return the raw paths to the data instead of parsing
        it. This method returns a context manager.
        """
        return DatasetBase._ReturnPathContext(self)

class DetectionDatasetBase(DatasetBase):
    """
    This class defines basic interface for object detection

    :param base_path: directory containing the zip files, or the required data
    :param inzip: whether the dataset is store in original zip archives or unzipped
    :param phase: training, validation or testing
    :param trainval_split: the ratio to split training dataset. See
                           documentation of :func:`split_trainval` for detail.
    :param trainval_random: whether select the train/val split randomly. See
                           documentation of :func:`split_trainval` for detail.
    """
    VALID_CAM_NAMES: list
    '''
    List of valid sensor names of camera
    '''

    VALID_LIDAR_NAMES: list
    '''
    List of valid sensor names of lidar
    '''

    VALID_OBJ_CLASSES: Enum
    '''
    List of valid object labels
    '''

    def __init__(self,
                 base_path: Union[str, Path],
                 inzip: bool = False,
                 phase: str = "training",
                 trainval_split: Union[float, List[int]] = 1.,
                 trainval_random: Union[bool, int, str] = False):
        super().__init__(base_path, inzip=inzip, phase=phase,
                         trainval_split=trainval_split, trainval_random=trainval_random)

    def lidar_data(self,
                   idx: Union[int, tuple],
                   names: Optional[Union[str, List[str]]] = None
                   ) -> Union[NdArray, List[NdArray]]:
        '''
        Return the lidar point cloud data

        :param names: name of requested lidar sensors. The default sensor is
                      the first element in :attr:`VALID_LIDAR_NAMES`.
        :param idx: index of requested lidar frames
        '''
        raise NotImplementedError("abstract function")

    def camera_data(self, idx: Union[int, tuple],
                    names: Optional[Union[str, List[str]]] = None
                    ) -> Union[Image, List[Image]]:
        '''
        Return the camera image data

        :param names: name of requested camera sensors. The default sensor is
                      the first element in :attr:`VALID_CAM_NAMES`.
        :param idx: index of requested image frames
        '''
        raise NotImplementedError("abstract function")

    def calibration_data(self, idx: Union[int, tuple],
            raw: Optional[bool] = None) -> Union[TransformSet, Any]:
        '''
        Return the calibration data

        :param idx: index of requested frame
        :param raw: if false, converted :class:`d3d.abstraction.TransformSet`
                    will be returned, otherwise raw data will be returned in
                    original format.
        '''
        raise NotImplementedError("abstract function")

    def annotation_3dobject(self, idx: Union[int, tuple],
            raw: Optional[bool] = None) -> Union[Target3DArray, Any]:
        '''
        Return list of converted ground truth targets in lidar frame.

        :param idx: index of requested frame
        :param raw: if false, targets will be converted to d3d
                    :class:`d3d.abstraction.Target3DArray` format,
                    otherwise raw data will be returned in original format
        '''
        raise NotImplementedError("abstract function")

    def identity(self, idx: int) -> tuple:
        '''
        Return something that can track the data back to original dataset. The result tuple can be passed
            to any accessors above and directly access given data.

        :param idx: index of requested frame to be parsed
        '''
        raise NotImplementedError("abstract function")

class TrackingDatasetBase(DetectionDatasetBase):
    """
    Tracking dataset is similarly defined with detection dataset. The two major differences are
    1. Tracking dataset use (sequence_id, frame_id) as identifier.
    2. Tracking dataset provide unique object id across time frames.

    :param base_path: directory containing the zip files, or the required data
    :param inzip: whether the dataset is store in original zip archives or unzipped
    :param phase: training, validation or testing
    :param trainval_split: the ratio to split training dataset. See
                           documentation of :func:`split_trainval` for detail.
    :param trainval_random: whether select the train/val split randomly. See
                           documentation of :func:`split_trainval` for detail.
    :param nframes: number of consecutive frames returned from the accessors

        * If it's a positive number, then it returns adjacent frames with total number reduced
        * If it's a negative number, absolute value of it is consumed
        * If it's zero, then it act like object detection dataset, which means the methods will return unpacked data
    :param trainval_byseq: Whether split trainval partitions by sequences instead of frames
    """

    def __init__(self,
                 base_path: Union[str, Path],
                 inzip: bool = False,
                 phase: str = "training",
                 trainval_split: Union[float, List[int]] = 1.,
                 trainval_random: Union[bool, int, str] = False,
                 trainval_byseq = False,
                 nframes: int = 0):
        super().__init__(base_path, inzip=inzip, phase=phase,
                         trainval_split=trainval_split, trainval_random=trainval_random)
        self.nframes = abs(nframes)

    def _locate_frame(self, idx: int) -> Tuple[Any, int]:
        '''
        Subclass should implement this function to convert overall index to (sequence_id, frame_idx) to support
        decorator :func:`expand_idx` and :func:`expand_idx_name`

        :return: (seq_id, frame_idx) where frame_idx is the index of starting frame
        '''
        raise NotImplementedError("_locate_frame is not implemented!")

    def lidar_data(self,
                   idx: Union[int, tuple],
                   names: Optional[Union[str, List[str]]] = None
                   ) -> Union[NdArray, List[NdArray], List[List[NdArray]]]:
        '''
        If multiple frames are requested, the results will be a list of list. Outer list
        corresponds to frame names and inner list corresponds to time sequence. So
        len(names) × len(frames) data objects will be returned

        :param names: name of requested lidar sensors. The default frame is
                      the first element in :attr:`VALID_LIDAR_NAMES`.
        :param idx: index of requested lidar frames

            * If single index is given, then the frame indexing is done on the whole dataset with trainval split
            * If a tuple is given, it's considered to be a unique id of the frame (from :meth:`identity` method),
              trainval split is ignored in this way and nframes offset is not added
        '''
        raise NotImplementedError("_locate_frame is not implemented!")

    def camera_data(self,
                    idx: Union[int, tuple],
                    names: Optional[Union[str, List[str]]] = None
                    ) -> Union[Image, List[Image], List[List[Image]]]:
        '''
        Return the camera image data

        :param names: name of requested camera sensors. The default sensor is
                      the first element in :attr:`VALID_CAM_NAMES`.
        :param idx: index of requested image frames, see description in
                    :meth:`lidar_data` method.
        '''
        raise NotImplementedError("abstract function")

    def calibration_data(self, idx: Union[int, tuple],
            raw: Optional[bool] = False) -> Union[TransformSet, Any]:
        '''
        Return the calibration data. Notices that we assume the calibration is
        fixed among one squence, so it always return a single object.

        :param idx: index of requested lidar frames
        :param raw: If false, converted :class:`d3d.abstraction.TransformSet`
                    will be returned, otherwise raw data will be returned in
                    original format
        '''
        raise NotImplementedError("abstract function")

    def intermediate_data(self, idx: Union[int, tuple],
            names: Optional[Union[str, List[str]]] = None,
            ninter_frames: int = 1) -> dict:
        '''
        Return the intermediate data (and annotations) between keyframes. For
        key frames data, please use corresponding function to load them

        :param idx: index of requested data frames
        :param names: name of requested sensors.
        :param ninter_frames: number of intermediate frames. If set to None,
                              then all frames will be returned.
        '''
        return []

    def annotation_3dobject(self,
                            idx: Union[int, tuple],
                            raw: Optional[bool] = False
        ) -> Union[Target3DArray, List[Target3DArray]]:
        '''
        Return list of converted ground truth targets in lidar frame.

        :param idx: index of requested frame
        :param raw: if false, targets will be converted to d3d
                    :class:`d3d.abstraction.Target3DArray` format, otherwise
                    raw data will be returned in original format.
        '''
        raise NotImplementedError("abstract function")

    def identity(self, idx: int) -> Union[tuple, List[tuple]]:
        '''
        Return something that can track the data back to original dataset

        :param idx: index of requested frame to be parsed
        :return: if :obj:`nframes` > 0, then the function return a list of ids
                 which are consistent with other functions.
        '''
        raise NotImplementedError("abstract function")

    def pose(self,
             idx: Union[int, tuple],
             raw: Optional[bool] = False,
             names: Optional[Union[str, List[str]]] = None
            ) -> Union[EgoPose, Any]:
        '''
        Return (relative) pose of the vehicle for the frame. The base frame should be ground attached
        which means the base frame will follow a East-North-Up axis order.

        :param idx: index of requested frame
        :param names: specify the sensor whose pose is requested. This option
                      only make sense when the dataset contains separate
                      timestamps for data from each sensor. In this case, the
                      pose either comes from dataset, or from interpolation.
        :param raw: if false, targets will be converted to d3d
                    :class:`d3d.abstraction.EgoPose` format, otherwise raw
                    data will be returned in original format.
        '''
        raise NotImplementedError("abstract function")

    def timestamp(self,
                  idx: Union[int, tuple],
                  names: Optional[Union[str, List[str]]] = None
                 ) -> Union[int, List[int]]:
        '''
        Return the timestamp of frame specified by the index, represented by
        Unix timestamp in macroseconds (usually 16 digits integer)

        :param idx: index of requested frame
        :param names: specify the sensor whose pose is requested. This option
                      only make sense when the dataset contains separate
                      timestamps for data from each sensor.
        '''
        raise NotImplementedError("abstract function")

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

    There is a parameter :obj:bypass` added to decorated function, which is used to call
    the original underlying method without expansion.
    '''
    @functools.wraps(func)
    def wrapper(self: TrackingDatasetBase, idx, bypass=False, **kwargs):
        if isinstance(idx, int):
            seq_id, frame_idx = self._locate_frame(idx)
        else:
            seq_id, frame_idx = idx

        if self.nframes == 0 or bypass:
            return func(self, (seq_id, frame_idx), **kwargs)
        else:
            return [func(self, (seq_id, idx), **kwargs)
                    for idx in range(frame_idx, frame_idx + self.nframes + 1)]

    return wrapper

def expand_name(valid_names: List[str]) -> Callable[[Callable], Callable]:
    '''
    This decorator works similar to expand_idx with support to distribute frame names.
    Note that this function acts as a decorator factory instead of decorator

    :param valid_names: List of valid sensor names
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

def expand_idx_name(valid_names: List[str]) -> Callable[[Callable], Callable]:
    '''
    This decorator works similar to expand_idx with support to distribute both indices and
    frame names. Note that this function acts as a decorator factory instead of decorator

    There is a parameter :obj:bypass` added to decorated function, which is used to call
    the original underlying method without expansion.

    :param valid_names: List of valid sensor names
    '''
    def decorator(func):
        default_names = inspect.signature(func).parameters["names"].default
        assert default_names != inspect.Parameter.empty, \
               "The decorated function should have default names value"

        @functools.wraps(func)
        def wrapper(self: TrackingDatasetBase, idx, names=default_names, bypass=False, **kwargs):
            if isinstance(idx, int):
                seq_id, frame_idx = self._locate_frame(idx)
            else:
                seq_id, frame_idx = idx
            unpack_result, names = check_frames(names, valid_names)

            results = []
            for name in names:
                if self.nframes == 0 or bypass:
                    results.append(func(self, (seq_id, frame_idx), names=name, **kwargs))
                else:
                    results.append([func(self, (seq_id, idx), names=name, **kwargs)
                                    for idx in range(frame_idx, frame_idx + self.nframes + 1)])
            return results[0] if unpack_result else results

        return wrapper

    return decorator

class NumberPool:
    '''
    This class is a utility for multiprocessing using tqdm, define the task as

    .. code-block:: python

        def task(ntqdm, ...):
            ...
            for data in tqdm(..., position=ntqdm, leave=False):
                ...

    Then the parallel progress bars will be displayed in place.

    :param processes: Number of processes available in the pool. If processes < 1,
                      then functions will be executed in current thread.
    :param offset: The offset added to the `ntqdm` value of all process. This is useful
                   when you want to display a progress bar in outer loop.
    '''

    def __init__(self,
                 processes: int,
                 offset: int = 0,
                 *args, **kargs):
        if processes == 0:
            self._single_thread = True
        else:
            self._single_thread = False
            self._ppool = Pool(processes, initializer=tqdm.set_lock, # pool of processes
                            initargs=(tqdm.get_lock(),), *args, **kargs)
            self._npool = Manager().Array('B', [0] * processes) # pool of position number
            self._nlock = Manager().Lock() # lock for self._npool
            self._nqueue = 0 # number of tasks in pool
            self._offset = offset
            self._complete_event = Event()

    @staticmethod
    def _wrap_func(func, args, pool, nlock, offset):
        n = -1
        with nlock:
            n = next(i for i, v in enumerate(pool) if v == 0)
            pool[n] = 1
        ret = func(n + offset, *args)
        return (n, ret)

    def apply_async(self, func, args=(), callback=None):
        if self._single_thread:
            result = func(0, *args)
            if callback is not None:
                callback(result)
            return result

        def _wrap_cb(ret):
            n, oret = ret
            with self._nlock:
                self._npool[n] = 0
            self._nqueue -= 1
            if callback is not None:
                callback(oret)
            self._complete_event.set()

        self._nqueue += 1
        self._ppool.apply_async(NumberPool._wrap_func,
                                (func, args, self._npool,
                                 self._nlock, self._offset),
                                callback=_wrap_cb,
                                error_callback=lambda e: print(
                                    f"{type(e).__name__}: {e}")
                                )

    def wait_for_once(self, margin: int = 0):
        """
        Block current thread and wait for one available process
        
        :param margin: Define when a process is available. The method will
                       return when there is :obj:`nprocess + margin` processes
                       in the pool.
        """
        if self._nqueue >= len(self._npool) + margin: # only wait if the pool is full
            self._complete_event.wait()
        self._complete_event.clear()

    def close(self):
        self._ppool.close()

    def join(self):
        self._ppool.join()

    # TODO: add imap function and support progress bar for imap

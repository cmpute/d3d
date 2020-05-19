import os.path as osp
from multiprocessing import Manager, Pool
from typing import Any, List, Optional, Union
from zipfile import ZipFile

import numpy as np
import numpy.random as npr
from numpy import ndarray as NdArray
from PIL.Image import Image
from tqdm import tqdm

from d3d.abstraction import ObjectTarget3DArray, TransformSet


class DetectionDatasetBase:
    VALID_CAM_NAMES: list
    VALID_LIDAR_NAMES: list

    def __init__(self, base_path, inzip=False, phase="training", trainval_split=1, trainval_random=False):
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
        raise NotImplementedError("This is a base class, should not be initialized!")

    def _split_trainval(self, phase, total_count, trainval_split, trainval_random):
        '''
        split frames for training or validation set
        '''
        if isinstance(trainval_split, list):
            self.frames = trainval_split
        else: # trainval_split is a number
            if isinstance(trainval_random, bool):
                self.frames = npr.default_rng().permutation(total_count) \
                    if trainval_random else np.arange(total_count)
            elif isinstance(trainval_random, int):
                gen = npr.default_rng(seed=trainval_random)
                self.frames = gen.permutation(total_count)
            elif trainval_random == "r": # predefined shuffle type
                self.frames = np.arange(total_count)[::-1]
            else:
                raise ValueError("Invalid trainval_random type!")

            if phase == 'training':
                self.frames = self.frames[:int(total_count * trainval_split)]
            elif phase == 'validation':
                self.frames = self.frames[int(total_count * trainval_split):]

    def lidar_data(self, idx: int, names:Optional[Union[str, List[str]]] = None, concat: bool = False) -> Union[NdArray, List[NdArray]]:
        '''
        :param names: name of requested lidar frames
        :param concat: whether to convert the point clouds to base frame and concat them.
                       If only one frame requested, the conversion to base frame will still be performed.
        '''
        pass

    def camera_data(self, idx: int, names: Optional[Union[str, List[str]]] = None) -> Union[Image, List[Image]]:
        pass

    def calibration_data(self, idx: int, raw: Optional[bool] = None) -> TransformSet:
        pass

    def lidar_label(self, idx: int) -> dict:
        pass

    def lidar_objects(self, idx: int) -> ObjectTarget3DArray:
        pass

    def identity(self, idx: int) -> Any:
        '''
        Return something that can track the data back to original dataset
        '''
        pass

def _check_frames(names, valid):
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

class ZipCache:
    '''
    This class is a utility for zip reading. It will retain the reference
    to the last accessed zip file. It also handles the close of the object
    '''
    def __init__(self, size=1):
        if size > 1:
            raise NotImplementedError()
        self._cache = None
        self._cache_path = None

    def open(self, path, **kvargs):
        path = osp.abspath(path)
        if path != self._cache_path:
            if self._cache is not None:
                self._cache.close()
            self._cache = ZipFile(path, **kvargs)
            self._cache_path = path
        return self._cache


def _wrap_func(func, args, pool, nlock, offset):
    n = -1
    with nlock:
        n = next(i for i,v in enumerate(pool) if v == 0)
        pool[n] = 1
    ret = func(n + offset, *args)
    return (n, ret)

class NumberPool:
    '''
    This class is a utility for multiprocessing using tqdm
    '''
    def __init__(self, processes, offset=0, *args, **kargs):
        self._ppool = Pool(processes, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), *args, **kargs)
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

        self._ppool.apply_async(_wrap_func,
            (func, args, self._npool, self._nlock, self._offset),
            callback=_wrap_cb,
            error_callback=lambda e: print(f"{type(e).__name__}: {e}")
        )

    def close(self):
        self._ppool.close()

    def join(self):
        self._ppool.join()

import os.path as osp
from typing import List, Union, Any, Optional
from zipfile import ZipFile

from numpy import ndarray as NdArray
from PIL.Image import Image

from d3d.abstraction import ObjectTarget3DArray, TransformSet


class DetectionDatasetBase:
    VALID_CAM_NAMES: list
    VALID_LIDAR_NAMES: list

    def __init__(self):
        raise NotImplementedError("This is a base class, should not be initialized!")

    def lidar_data(self, idx: int, names:Optional[Union[str, List[str]]] = None) -> Union[NdArray, List[NdArray]]:
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

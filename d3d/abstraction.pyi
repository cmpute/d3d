import io
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Tuple, Type, Union

import numpy
import scipy.spatial.transform
import torch


class ObjectTag:
    def __init__(self,
                 labels: Union[Union[Enum, str, int], Iterable[Union[Enum, str, int]]],
                 mapping: Enum = None,
                 scores: Union[float, Iterable[float]] = None):
        ...

    mapping: Type[Enum]
    labels: List[int]
    scores: List[float]

    def serialize(self) -> tuple: ...
    @classmethod
    def deserialize(cls, data: tuple) -> ObjectTag: ...

class ObjectTarget3D:
    def __init__(self,
                 position: Union[numpy.ndarray, List[float]],
                 orientation: Union[numpy.ndarray, List[float],
                                    scipy.spatial.transform.Rotation],
                 dimension: Union[numpy.ndarray, List[float]],
                 tag: ObjectTag,
                 tid: int = 0,
                 position_var: Union[numpy.ndarray, List[float]] = None,
                 orientation_var: Union[numpy.ndarray, List[float]] = None,
                 dimension_var: Union[numpy.ndarray, List[float]] = None):
        ...

    @property
    def position(self) -> numpy.ndarray: ...
    @property
    def position_var(self) -> numpy.ndarray: ...
    @property
    def dimension(self) -> numpy.ndarray: ...
    @property
    def dimension_var(self) -> numpy.ndarray: ...
    @property
    def orientation(self) -> scipy.spatial.transform.Rotation: ...
    @property
    def orientation_var(self) -> float: ...
    @property
    def tid(self) -> int: ...
    @property
    def tag(self) -> ObjectTag: ...
    @property
    def tag_top(self) -> Enum: ...
    @property
    def tag_name(self) -> str: ...
    @property
    def tag_score(self) -> float: ...
    @property
    def yaw(self) -> float: ...
    @property
    def corners(self) -> numpy.ndarray: ...
    @property
    def tid64(self) -> str: ...

    def to_numpy(self, box_type: str = "ground") -> numpy.ndarray: ...
    def serialize(self) -> tuple: ...
    @classmethod
    def deserialize(cls, data: tuple) -> "ObjectTarget3D": ...

class TrackingTarget3D(ObjectTarget3D):
    def __init__(self,
                 position: Union[numpy.ndarray, List[float]],
                 orientation: Union[numpy.ndarray, List[float],
                                    scipy.spatial.transform.Rotation],
                 dimension: Union[numpy.ndarray, List[float]],
                 velocity: Union[numpy.ndarray, List[float]],
                 angular_velocity: Union[numpy.ndarray, List[float]],
                 tag: ObjectTag,
                 tid: int = 0,
                 position_var: Union[numpy.ndarray, List[float]] = None,
                 orientation_var: Union[numpy.ndarray, List[float]] = None,
                 dimension_var: Union[numpy.ndarray, List[float]] = None,
                 velocity_var: Union[numpy.ndarray, List[float]] = None,
                 angular_velocity_var: Union[numpy.ndarray, List[float]] = None,
                 history: float = None):
        ...

    @property
    def velocity(self) -> numpy.ndarray: ...
    @property
    def velocity_var(self) -> numpy.ndarray: ...
    @property
    def angular_velocity(self) -> numpy.ndarray: ...
    @property
    def angular_velocity_var(self) -> numpy.ndarray: ...
    @property
    def history(self) -> float: ...

    def to_numpy(self, box_type: str = "ground") -> numpy.ndarray: ...
    def serialize(self) -> tuple: ...
    @classmethod
    def deserialize(cls, data: tuple) -> "TrackingTarget3D": ...

class Target3DArray(list):
    def __init__(self,
                 iterable: List[Union[ObjectTarget3D, TrackingTarget3D]] = [],
                 frame: str = None,
                 timestamp: int = 0): ...

    @property
    def frame(self) -> str: ...
    @property
    def timestamp(self) -> int: ...

    def to_numpy(self, box_type: str = "ground") -> numpy.ndarray: ...
    def to_torch(self, box_type: str = "ground") -> torch.Tensor: ...
    def serialize(self) -> tuple: ...
    @classmethod
    def deserialize(cls, data: tuple) -> ObjectTag: ...
    def dump(self, output: Union[str, Path, io.BufferedIOBase]): ...
    @classmethod
    def load(cls, input: Union[str, Path, io.BufferedIOBase]) -> Target3DArray: ...
    def filter_tag(self, tags: Union[str, Enum, List[str], List[Enum]]) -> Target3DArray: ...
    def filter_score(self, score: float) -> Target3DArray: ...
    def filter_position(self, x_min: float = float('nan'),
                        x_max: float = float('nan'),
                        y_min: float = float('nan'),
                        y_max: float = float('nan'),
                        z_min: float = float('nan'),
                        z_max: float = float('nan')) -> Target3DArray: ...

class EgoPose:
    def __init__(self,
                 position: Union[numpy.ndarray, List[float]],
                 orientation: Union[numpy.ndarray, List[float],
                                    scipy.spatial.transform.Rotation],
                 position_var: Union[numpy.ndarray, List[float]] = None,
                 orientation_var: Union[numpy.ndarray, List[float]] = None):
        ...

    @property
    def position(self) -> numpy.ndarray: ...
    @property
    def orientation(self) -> scipy.spatial.transform.Rotation: ...
    @property
    def position_var(self) -> numpy.ndarray: ...
    @property
    def orientation_var(self) -> numpy.ndarray: ...

    def homo(self) -> numpy.ndarray: ...

class CameraMetadata:
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def distort_coeffs(self) -> numpy.ndarray: ...
    @property
    def intri_matrix(self) -> numpy.ndarray: ...
    @property
    def mirror_coeff(self) -> float: ...

class LidarMetadata:
    pass

class RadarMetadata:
    pass

class PinMetadata:
    @property
    def lon(self) -> float: ...
    @property
    def lat(self) -> float: ...

class TransformSet:
    def __init__(self, base_frame: str): ...

    def set_intrinsic_general(self, frame_id: str,
                              metadata: Union[CameraMetadata, LidarMetadata,
                                              RadarMetadata, PinMetadata]): ...

    def set_intrinsic_camera(self, frame_id: str,
                             transform: numpy.ndarray,
                             size: Tuple[int, int],
                             rotate: bool,
                             distort_coeffs: Union[numpy.ndarray, List[float]],
                             intri_matrix: numpy.ndarray,
                             mirror_coeff: float): ...

    def set_intrinsic_lidar(self, frame_id: str): ...

    def set_intrinsic_radar(self, frame_id: str): ...

    def set_intrinsic_pinhole(self, frame_id: str,
                              size: Tuple[int, int],
                              cx: float,
                              cy: float,
                              fx: float,
                              fy: float,
                              s: float,
                              distort_coeffs: Union[numpy.ndarray, List[float]]): ...

    def set_intrinsic_map_pin(self, frame_id: str,
                              lon: float, lat: float): ...

    def set_extrinsic(self,
                      transform: numpy.ndarray,
                      frame_to: str,
                      frame_from: str): ...

    def get_extrinsic(self, frame_to: str, frame_from: str): ...

    def transform_objects(self, objects: Target3DArray, frame_to: str) -> Target3DArray: ...

    def transform_points(self, points: numpy.ndarray,
                         frame_to: str, frame_from: str) -> numpy.ndarray: ...

    def project_points_to_camera(self,
                                 points: numpy.ndarray,
                                 frame_from: str,
                                 frame_to: str,
                                 remove_outlier,
                                 return_dmask): ...

    def dump(self, output: Union[str, Path, io.BufferedIOBase]): ...
    @classmethod
    def load(cls, file: Union[str, Path, io.BufferedIOBase]) -> "TransformSet": ...

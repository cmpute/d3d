import unittest

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation

from d3d.abstraction import (ObjectTag, ObjectTarget3D, Target3DArray,
                             TrackingTarget3D)
from d3d.dataset.kitti import KittiObjectClass


class TestAbstraction(unittest.TestCase):
    def test_dump_and_load(self):
        obj_arr = Target3DArray(frame="someframe", timestamp=1.2345)
        track_arr = Target3DArray(frame="fixed", timestamp=0.1234)

        # add targets
        for i in range(10):
            position = np.array([i] * 3)
            position_var = np.diag(position)
            dimension = np.array([i] * 3)
            dimension_var = np.diag(position)
            orientation = Rotation.from_euler("Z", i)
            tid = "test%d" % i
            tag = ObjectTag(KittiObjectClass.Car, KittiObjectClass, 0.9)
            obj = ObjectTarget3D(position, orientation, dimension, tag, tid,
                position_var=position_var, dimension_var=dimension_var)
            obj_arr.append(obj)

            velocity = np.random.rand(3)
            velocity_var = np.random.rand(3, 3)
            avel = np.random.rand(3)
            avel_var = np.random.rand(3, 3)
            history = i * 0.1
            track = TrackingTarget3D(position, orientation, dimension, velocity, avel, tag,
                tid=tid, position_var=position_var, dimension_var=dimension_var, velocity_var=velocity_var,
                angular_velocity_var=avel_var, history=history)
            track_arr.append(track)
        
        data = msgpack.packb(obj_arr.serialize(), use_single_float=True)
        obj_arr_copy = Target3DArray.deserialize(msgpack.unpackb(data))

        assert len(obj_arr_copy) == len(obj_arr)
        assert obj_arr_copy.frame == obj_arr.frame
        assert obj_arr_copy.timestamp == obj_arr.timestamp
        for i in range(10):
            assert np.allclose(obj_arr_copy[i].position, obj_arr[i].position)
            assert np.allclose(obj_arr_copy[i].position_var, obj_arr[i].position_var)
            assert np.allclose(obj_arr_copy[i].dimension, obj_arr[i].dimension)
            assert np.allclose(obj_arr_copy[i].dimension_var, obj_arr[i].dimension_var)
            assert np.allclose(obj_arr_copy[i].orientation.as_quat(), obj_arr[i].orientation.as_quat())
            assert obj_arr_copy[i].tid == obj_arr[i].tid

            assert obj_arr_copy[i].tag.mapping == obj_arr[i].tag.mapping
            assert obj_arr_copy[i].tag.labels == obj_arr[i].tag.labels
            assert np.allclose(obj_arr_copy[i].tag.scores, obj_arr[i].tag.scores)

        data = msgpack.packb(track_arr.serialize(), use_single_float=True)
        track_arr_copy = Target3DArray.deserialize(msgpack.unpackb(data))

        assert len(track_arr_copy) == len(track_arr)
        assert track_arr_copy.frame == track_arr.frame
        assert track_arr_copy.timestamp == track_arr.timestamp
        for i in range(10):
            assert np.allclose(track_arr_copy[i].position, track_arr[i].position)
            assert np.allclose(track_arr_copy[i].position_var, track_arr[i].position_var)
            assert np.allclose(track_arr_copy[i].dimension, track_arr[i].dimension)
            assert np.allclose(track_arr_copy[i].dimension_var, track_arr[i].dimension_var)
            assert np.allclose(track_arr_copy[i].orientation.as_quat(), track_arr[i].orientation.as_quat())
            assert np.allclose(track_arr_copy[i].velocity, track_arr[i].velocity)
            assert np.allclose(track_arr_copy[i].velocity_var, track_arr[i].velocity_var)
            assert np.allclose(track_arr_copy[i].angular_velocity, track_arr[i].angular_velocity)
            assert np.allclose(track_arr_copy[i].angular_velocity_var, track_arr[i].angular_velocity_var)
            assert track_arr_copy[i].tid == track_arr[i].tid
            assert track_arr_copy[i].history == track_arr[i].history

            assert track_arr_copy[i].tag.mapping == track_arr[i].tag.mapping
            assert track_arr_copy[i].tag.labels == track_arr[i].tag.labels
            assert np.allclose(track_arr_copy[i].tag.scores, track_arr[i].tag.scores)

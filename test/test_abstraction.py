import unittest
import numpy as np
import msgpack

from scipy.spatial.transform import Rotation
from d3d.dataset.kitti import KittiObjectClass
from d3d.abstraction import ObjectTarget3DArray, ObjectTarget3D, ObjectTag

class TestAbstraction(unittest.TestCase):
    def test_dump_and_load(self):
        arr = ObjectTarget3DArray(frame="something")
        for i in range(10):
            position = np.array([i] * 3)
            position_var = np.diag(position)
            dimension = np.array([i] * 3)
            dimension_var = np.diag(position)
            orientation = Rotation.from_euler("Z", i)
            id_ = "test%d" % i
            tag = ObjectTag(KittiObjectClass.Car, KittiObjectClass, 0.9)
            obj = ObjectTarget3D(position, orientation, dimension, tag, id_,
                position_var=position_var, dimension_var=dimension_var)
            arr.append(obj)
        
        data = msgpack.packb(arr.serialize(), use_single_float=True)
        arr_copy = ObjectTarget3DArray.deserialize(msgpack.unpackb(data))

        assert len(arr_copy) == len(arr)
        assert arr_copy.frame == arr.frame
        for i in range(10):
            assert np.allclose(arr_copy[i].position, arr[i].position)
            assert np.allclose(arr_copy[i].position_var, arr[i].position_var)
            assert np.allclose(arr_copy[i].dimension, arr[i].dimension)
            assert np.allclose(arr_copy[i].dimension_var, arr[i].dimension_var)
            assert np.allclose(arr_copy[i].orientation.as_quat(), arr[i].orientation.as_quat())
            assert arr_copy[i].id == arr[i].id

            assert arr_copy[i].tag.mapping == arr[i].tag.mapping
            assert arr_copy[i].tag.labels == arr[i].tag.labels
            assert np.allclose(arr_copy[i].tag.scores, arr[i].tag.scores)

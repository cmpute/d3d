import unittest

import numpy as np
import scipy.special as sps
import torch

from d3d.math import i0e_cc, i0e_cuda, i1e_cc, i1e_cuda

class TestMathModule(unittest.TestCase):
    def test_i0e(self):
        x = torch.rand(100, 100) * 10 - 5
        gt = sps.i0e(x.numpy())
        assert np.allclose(i0e_cc(x).numpy(), gt)
        assert np.allclose(i0e_cuda(x.cuda()).cpu().numpy(), gt)

    def test_i1e(self):
        x = torch.rand(100, 100) * 10 - 5
        gt = sps.i1e(x.numpy())
        assert np.allclose(i1e_cc(x).numpy(), gt)
        assert np.allclose(i1e_cuda(x.cuda()).cpu().numpy(), gt)

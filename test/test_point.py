import unittest

import numpy as np
import torch

from d3d.point import aligned_scatter


class TestPointModule(unittest.TestCase):
    def test_aligned_scatter_forward(self):
        coord = torch.tensor([[0, 0.25, 0.25, 0.25], [1, 1.25, 1.25, 1.25], [0, 2.25, 2.25, 2.25]])
        image_feat = torch.rand(2, 10, 3, 3, 3)
        indexing = lambda icoord: (icoord[:, 0], slice(None)) + tuple(icoord[:, i] for i in range(1, coord.shape[1]))

        lcoords = np.array(np.meshgrid([0,1], [0,1], [0,1]))
        lcoords = torch.tensor(lcoords.T.reshape(-1, 3))

        # test drop
        icoord = coord.long()
        pfeat = aligned_scatter(coord, image_feat, "drop")
        assert torch.allclose(pfeat, image_feat[indexing(icoord)])

        # test mean
        pfeat = aligned_scatter(coord, image_feat, "mean")

        icoord = torch.cat([torch.full((8,1), 0, dtype=torch.long), lcoords], dim=1)
        assert torch.allclose(pfeat[0], torch.mean(image_feat[indexing(icoord)], dim=0))
        icoord = torch.cat([torch.full((8,1), 1, dtype=torch.long), lcoords+1], dim=1)
        assert torch.allclose(pfeat[1], torch.mean(image_feat[indexing(icoord)], dim=0))
        assert torch.allclose(pfeat[2], image_feat[0, :, 2, 2, 2])

        # test linear
        pfeat = aligned_scatter(coord, image_feat, "linear")
        nhigh = torch.sum(lcoords, dim=1).long()
        wmap = torch.tensor([0.25**i * 0.75**(3-i) for i in range(4)])
        lweight = wmap[nhigh]

        icoord = torch.cat([torch.full((8,1), 0, dtype=torch.long), lcoords], dim=1)
        assert torch.allclose(pfeat[0], torch.sum(image_feat[indexing(icoord)] * lweight.unsqueeze(1), dim=0))
        icoord = torch.cat([torch.full((8,1), 1, dtype=torch.long), lcoords+1], dim=1)
        assert torch.allclose(pfeat[1], torch.sum(image_feat[indexing(icoord)] * lweight.unsqueeze(1), dim=0))
        assert torch.allclose(pfeat[2], image_feat[0, :, 2, 2, 2])

        # test max
        # pfeat = aligned_scatter(coord, image_feat, "max")

        # icoord = torch.cat([torch.full((8,1), 0, dtype=torch.long), lcoords], dim=1)
        # assert torch.allclose(pfeat[0], torch.max(image_feat[indexing(icoord)], dim=0).values)
        # icoord = torch.cat([torch.full((8,1), 1, dtype=torch.long), lcoords+1], dim=1)
        # assert torch.allclose(pfeat[1], torch.max(image_feat[indexing(icoord)], dim=0).values)
        # assert torch.allclose(pfeat[2], image_feat[0, :, 2, 2, 2])

    def test_aligned_scatter_forward_cuda(self):
        device = torch.device("cuda")
        coord = torch.tensor([[0, 0.25, 0.25, 0.25], [1, 1.25, 1.25, 1.25], [0, 2.25, 2.25, 2.25]]).cuda()
        image_feat = torch.rand(2, 10, 3, 3, 3).cuda()
        indexing = lambda icoord: (icoord[:, 0], slice(None)) + tuple(icoord[:, i] for i in range(1, coord.shape[1]))

        lcoords = np.array(np.meshgrid([0,1], [0,1], [0,1]))
        lcoords = torch.tensor(lcoords.T.reshape(-1, 3), device=device)

        # test drop
        icoord = coord.long()
        pfeat = aligned_scatter(coord, image_feat, "drop")
        assert torch.allclose(pfeat, image_feat[indexing(icoord)])

        # test mean
        pfeat = aligned_scatter(coord, image_feat, "mean")

        icoord = torch.cat([torch.full((8,1), 0, dtype=torch.long, device=device), lcoords], dim=1)
        assert torch.allclose(pfeat[0], torch.mean(image_feat[indexing(icoord)], dim=0))
        icoord = torch.cat([torch.full((8,1), 1, dtype=torch.long, device=device), lcoords+1], dim=1)
        assert torch.allclose(pfeat[1], torch.mean(image_feat[indexing(icoord)], dim=0))
        assert torch.allclose(pfeat[2], image_feat[0, :, 2, 2, 2])

        # test linear
        pfeat = aligned_scatter(coord, image_feat, "linear")
        nhigh = torch.sum(lcoords, dim=1).long().cuda()
        wmap = torch.tensor([0.25**i * 0.75**(3-i) for i in range(4)], device=device)
        lweight = wmap[nhigh]

        icoord = torch.cat([torch.full((8,1), 0, dtype=torch.long, device=device), lcoords], dim=1)
        assert torch.allclose(pfeat[0], torch.sum(image_feat[indexing(icoord)] * lweight.unsqueeze(1), dim=0))
        icoord = torch.cat([torch.full((8,1), 1, dtype=torch.long, device=device), lcoords+1], dim=1)
        assert torch.allclose(pfeat[1], torch.sum(image_feat[indexing(icoord)] * lweight.unsqueeze(1), dim=0))
        assert torch.allclose(pfeat[2], image_feat[0, :, 2, 2, 2])

if __name__ == "__main__":
    TestPointModule().test_aligned_scatter_forward_cuda()

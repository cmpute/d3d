import torch

from .point_impl import aligned_scatter_forward, aligned_scatter_forward_cuda, AlignType

class AlignedScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, image_feature):
        ctx.save_for_backward(coords)

    @staticmethod
    def backward(ctx, grad):
        coords = ctx.saved_variables

def aligned_scatter(coordinates, feature_map, method="drop"):
    '''
    Gather the values given coordinates in feature_map.
    
    :param feature_map: should have shape B x C x D1 x D2... x Dm, the number of remaining dimensions (m) should be consistent with coordinate dimension
    :param coordinates: should have shape N x (m+1), it should contains batch index at the first dimension
    :param method:
        drop - directly convert decimal coordinates to integers
        mean - mean of surrounding values
        linear - (bi)linear interpolation of surrounding values
        max - maximum of surrounding values
        nearest - value of nearest value
    :return: extracted features with shape N x C

    Note: right now only `drop`, `mean` and `linear` are implemented
    '''
    method = (method or "DROP").upper()
    if method == "DROP":
        coordinates = coordinates.long()
        _, ndim = coordinates.shape
        assert len(feature_map.shape) == ndim + 1

        indexing = (coordinates[:, 0], slice(None)) + tuple(coordinates[:, i] for i in range(1, ndim))
        return feature_map[indexing]
    else:
        align_type = getattr(AlignType, method)
        if feature_map.is_cuda:
            return aligned_scatter_forward_cuda(coordinates, feature_map, align_type)
        else:
            return aligned_scatter_forward(coordinates, feature_map, align_type)

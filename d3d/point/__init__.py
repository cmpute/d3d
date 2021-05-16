try:
    import torch
    from .point_impl import (cuda_available,
        AlignType, aligned_scatter_forward, aligned_scatter_backward)
except ImportError:
    raise ImportError("Cannot find compiled library! D3D is probably compiled without pytorch!")

if cuda_available:
    from .point_impl import (aligned_scatter_backward_cuda,
        aligned_scatter_forward_cuda)


class AlignedScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image_feature, coords, atype):
        ctx.save_for_backward(coords)
        ctx.atype = atype
        ctx.image_shape = image_feature.shape
        ctx.image_dtype = image_feature.dtype
        ctx.image_device = image_feature.device

        if image_feature.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            return aligned_scatter_forward_cuda(coords, image_feature, atype)
        else:
            return aligned_scatter_forward(coords, image_feature, atype)

    @staticmethod
    def backward(ctx, grad):
        coords, = ctx.saved_tensors

        image_grad = torch.zeros(ctx.image_shape, dtype=ctx.image_dtype, device=ctx.image_device)
        if grad.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            aligned_scatter_backward_cuda(coords, grad, ctx.atype, image_grad)
        else:
            aligned_scatter_backward(coords, grad, ctx.atype, image_grad)
        return image_grad, None, None


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
        return AlignedScatter.apply(feature_map, coordinates, align_type)

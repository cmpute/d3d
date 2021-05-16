try:
    import torch
    from .math_impl import (cuda_available,
        i0e as i0e_cc, i1e as i1e_cc)
except ImportError:
    raise ImportError("Cannot find compiled library! D3D is probably compiled without pytorch!")

if cuda_available:
    from .math_impl import i0e_cuda, i1e_cuda

class I0Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            return i0e_cuda(x)
        return i0e_cc(x)

    @staticmethod
    def backward(ctx, grad):
        if grad.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            return i1e_cuda(grad)
        return i1e_cc(grad)

def i0e(x):
    '''
    Pytorch Autograd Function of
    `modified Bessel function <https://en.wikipedia.org/wiki/Bessel_function>`_
    with order 0
    '''
    return I0Exp.apply(x)

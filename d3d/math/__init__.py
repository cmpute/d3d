import torch

from .math_impl import i0e as i0e_cc, i0e_cuda, i1e as i1e_cc, i1e_cuda

class I0Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return i0e_cuda(x) if x.is_cuda else i0e_cc(x)

    @staticmethod
    def backward(ctx, grad):
        return i1e_cuda(grad) if grad.is_cuda else i1e_cc(grad)

def i0e(x):
    '''
    Pytorch Autograd Function of modified Bessel function with order 0
    '''
    return I0Exp.apply(x)

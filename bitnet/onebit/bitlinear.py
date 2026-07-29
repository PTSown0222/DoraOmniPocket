import torch
import torch.nn as nn
import torch.nn.functional as F

from bitnet.weight_quant import weight_quant_triton
from bitnet.activation_quant import activation_quant_triton

class WeightQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        return weight_quant_triton(weight)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class ActivationQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return activation_quant_triton(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = WeightQuantSTE.apply(self.weight)
        x_q = ActivationQuantSTE.apply(x)
        return F.linear(x_q, w_q, self.bias)
import torch
import torch.nn as nn
import torch.nn.functional as F
from bitnet.weight_quant import weight_quant_triton
from bitnet.activation_quant import activation_quant_triton


# def activation_quant(x: Tensor):
#     """Per token quantization to 8bits. No grouping is needed for quantization

#     Args:
#         x (Tensor): _description_

#     Returns:
#         _type_: _description_
#     """
#     scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
#     y = (x * scale).round().clamp_(-128, 127) / scale
#     return y

# def weight_quant(w: Tensor):
#     scale = w.abs().mean()
#     e = w.mean()
#     u = (w - e).sign() * scale
#     return u

class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim) / self.scale)

    def forward(self, x):
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma

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

class BitLinear(nn.Linear):
    """
    Custom linear layer with bit quantization.

    Args:
        dim (int): The input dimension of the layer.
        training (bool, optional): Whether the layer is in training mode or not. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the layer.

    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        w = self.weight
        x_norm = RMSNorm(self.in_features)(x)

        # STE using detach
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y
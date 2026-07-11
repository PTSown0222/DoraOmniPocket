# my_custom_llm/modules/normalization.py
"""
module implements various normalization techniques for transforming input data.
"""
import torch
import torch.nn as nn

# Pre-Norm (RMSNorm)
class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6):
        """
        Root Mean Square Layer Normalization (RMSNorm).
        Based on Llama/Gemma.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def _norm(self, x):
        # RMS normalization formula:
        # x = x / sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # avoid overflow by normalizing in float32
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Post-Norm (LayerNorm)
class LayerNorm(nn.Module):
    """
    Standard Layer Normalization
    """
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim)) # Scale (gamma)
        self.bias = nn.Parameter(torch.zeros(emb_dim))  # Shift (beta)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * norm_x + self.bias
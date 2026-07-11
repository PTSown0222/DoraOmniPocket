import torch
import torch.nn as nn
from torch import Tensor

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron with one hidden layer.

    This module is used within the Transformer block for feed-forward processing.
    It expands the input embedding size, applies a ReLU activation, and then projects it back
    to the original embedding size.

    Args:
        n_embed (int): The dimensionality of the input embedding.
    """
    def __init__(self, n_embed: int):
        super().__init__(self):
        
import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    The feed-forward network (FFN) in a transformer block.

    This specific implementation uses a SwiGLU-like architecture (using GELU instead of Swish),
    which is common in modern transformers like Llama and Gemma. It involves three linear
    projections and a gated activation.
    """

    def __init__(self, cfg: dict):
        """
        Initializes the FeedForward network.

        Args:
            cfg (dict): A configuration dictionary containing model parameters like
                        'emb_dim', 'hidden_dim', and 'dtype'.
        """
        super().__init__()
        # The first linear layer, often called the "up" projection, expands the dimension.
        self.fc1 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        # The second linear layer also expands the dimension and acts as the "gate" in the SwiGLU-like structure.
        self.fc2 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        # The third linear layer, the "down" projection, maps the dimension back to the original embedding size.
        self.fc3 = nn.Linear(
            cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FeedForward network.

        The computation is: `Output = W3(GELU(W1 * x) * (W2 * x))`
        """
        # Apply the first and second linear projections to the input.
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        # Apply the GELU activation function to the first projection.
        # The 'tanh' approximation is a faster variant of GELU.
        activated_x = nn.functional.gelu(x_fc1, approximate="tanh")
        # Perform element-wise multiplication (the "gating" mechanism).
        x = activated_x * x_fc2
        # Apply the final "down" projection.
        return self.fc3(x)


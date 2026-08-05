 
import gc
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# ------- Model Size Calculation ------- #
def calculate_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    total_params =  total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params:,}")
    
    # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    total_size_bytes = total_params * 4
    
    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    print(f"Total size of the model: {total_size_mb:.2f} MB")


def start_memory_tracking():
    """Initialize GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        print("This notebook is intended for CUDA GPUs but CUDA is not available.")

def print_memory_usage():
    max_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    print(f"Maximum GPU memory allocated: {max_gpu_memory:.1f} GB")


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3) 
    torch.cuda.reset_peak_memory_stats()
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"Maximum GPU memory allocated: {max_memory_allocated:.1f} GB")


# ------- Plot Attention Weights ------- #
class SelfAttention(torch.nn.Module):
    """
    A self-attention module that takes an input tensor and returns a tensor with self-attention applied.

    Args:
        d_model (int): The input dimensionality.

    Attributes:
        d_model (int): The input dimensionality.
        q_linear (torch.nn.Linear): A linear layer applied to the query.
        k_linear (torch.nn.Linear): A linear layer applied to the key.
        v_linear (torch.nn.Linear): A linear layer applied to the value.

    Methods:
        forward(x): Forward pass of the self-attention module.

    Returns:
        torch.Tensor: The output tensor of the self-attention module.
        torch.Tensor: The attention weights.

    """
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model)), dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output, attn_weights


class SelfAttentionModel(torch.nn.Module):
    """
    A self-attention model that applies self-attention to the input tensor and returns a tensor of output size.

    Args:
        d_model (int): The input dimensionality.
        output_size (int): The output dimensionality.

    Attributes:
        self_attn (SelfAttention): A self-attention module applied to the input.
        fc (torch.nn.Linear): A linear layer applied to the output of the self-attention module.

    Methods:
        forward(x): Forward pass of the self-attention model.

    Returns:
        torch.Tensor: The output tensor of the self-attention model.
        torch.Tensor: The attention weights.

    """
    def __init__(self, d_model, output_size):
        super(SelfAttentionModel, self).__init__()
        self.self_attn = SelfAttention(d_model)
        self.fc = torch.nn.Linear(d_model, output_size)
    
    def forward(self, x):
        x, attn_weights = self.self_attn(x)
        x = self.fc(x.mean(dim=1))
        return x, attn_weights

# plot attention weight
def plot_attention_weights(attn_weights, tokens):
    """
    Plots the attention weights of a self-attention model given a list of tokens.

    Args:
        attn_weights (torch.Tensor): A tensor of shape (batch_size, num_heads, seq_length, seq_length)
            representing the attention weights of a self-attention model.
        tokens (List[str]): A list of strings representing the tokens in the input sequence.

    Returns:
        None. This function does not return any value. It displays a plot of the attention weights.

    Example:    tokens = ['The', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
                attn_weights = torch.randn(1, 8, len(tokens), len(tokens))
                plot_attention_weights(attn_weights, tokens)

    """
    fig, ax = plt.subplots()
    im = ax.imshow(attn_weights.squeeze(0).detach().numpy(), cmap='GnBu')

    # Set ticks and labels
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)

    # Rotate tick labels and set label for color bar
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cbar = ax.figure.colorbar(im, ax=ax)

    # Add title and show plot
    ax.set_title("Self-Attention Weights")
    fig.tight_layout()

# ---- Usage ---- #

# initialize the sentence
sentence = "the movie was not bad"
tokens = sentence.split()

# Create word embeddings
embeddings = torch.nn.Embedding(len(tokens), 8)
inputs = embeddings(torch.tensor([i for i in range(len(tokens))]).long())


# Pass through self-attention model
model = SelfAttentionModel(d_model=8, output_size=1)
out, attn_weights = model(inputs.unsqueeze(0))

# Plot attention weights
plot_attention_weights(attn_weights, tokens)
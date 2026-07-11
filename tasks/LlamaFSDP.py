import dataclasses
import functools
import os

import datasets
import tokenizers
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.checkpoint import load, save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler


# Build the model
@dataclasses.dataclass
class LlamaConfig:
    """Define Llama model hyperparameters."""
    vocab_size: int = 50000  # Size of the tokenizer vocabulary
    max_position_embeddings: int = 2048  # Maximum sequence length
    hidden_size: int = 768  # Dimension of hidden layers
    intermediate_size: int = 4*768  # Dimension of MLP's hidden layer
    num_hidden_layers: int = 12  # Number of transformer layers
    num_attention_heads: int = 12  # Number of attention heads
    num_key_value_heads: int = 3  # Number of key-value heads for GQA


class RotaryPositionEncoding(nn.Module):
    """Rotary position encoding."""

    def __init__(self, dim: int, max_position_embeddings: int) -> None:
        """Initialize the RotaryPositionEncoding module.

        Args:
            dim: The hidden dimension of the input tensor to which RoPE is applied
            max_position_embeddings: The maximum sequence length of the input tensor
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        # compute a matrix of n\theta_i
        N = 10_000.0
        inv_freq = 1.0 / (N ** (torch.arange(0, dim, 2) / dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(max_position_embeddings)
        sinusoid_inp = torch.outer(position, inv_freq)
        # save cosine and sine matrices as buffers, not parameters
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def forward(self, x: Tensor) -> Tensor:
        """Apply RoPE to tensor x.

        Args:
            x: Input tensor of shape (batch_size, seq_length, num_heads, head_dim)

        Returns:
            Output tensor of shape (batch_size, seq_length, num_heads, head_dim)
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        device = x.device
        dtype = x.dtype
        # transform the cosine and sine matrices to 4D tensor and the same dtype as x
        cos = self.cos.to(device, dtype)[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin.to(device, dtype)[:seq_len].view(1, seq_len, 1, -1)
        # apply RoPE to x
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        output = (x * cos) + (rotated * sin)
        return output


class LlamaAttention(nn.Module):
    """Grouped-query attention with rotary embeddings."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads  # GQA: H_kv < H_q

        # hidden_size must be divisible by num_heads
        assert (self.head_dim * self.num_heads) == self.hidden_size

        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def reset_parameters(self):
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.o_proj.reset_parameters()

    def forward(self, hidden_states: Tensor, rope: RotaryPositionEncoding, attn_mask: Tensor) -> Tensor:
        bs, seq_len, dim = hidden_states.size()

        # Project inputs to Q, K, V
        query_states = self.q_proj(hidden_states).view(bs, seq_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bs, seq_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bs, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary position embeddings
        query_states = rope(query_states)
        key_states = rope(key_states)

        # Transpose tensors from BSHD to BHSD dimension for scaled_dot_product_attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Use PyTorch's optimized attention implementation
        # setting is_causal=True is incompatible with setting explicit attention mask
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=0.0,
            enable_gqa=True,
        )

        # Transpose output tensor from BHSD to BSHD dimension, reshape to 3D, and then project output
        attn_output = attn_output.transpose(1, 2).reshape(bs, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaMLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        # Two parallel projections for SwiGLU
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = F.silu  # SwiGLU activation function
        # Project back to hidden size
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def reset_parameters(self):
        self.gate_proj.reset_parameters()
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU activation: multiply gate and up-projected inputs
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaDecoderLayer(nn.Module):
    """Single transformer layer for a Llama model."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.mlp = LlamaMLP(config)

    def reset_parameters(self):
        self.input_layernorm.reset_parameters()
        self.self_attn.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, hidden_states: Tensor, rope: RotaryPositionEncoding, attn_mask: Tensor) -> Tensor:
        # First residual block: Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(hidden_states, rope=rope, attn_mask=attn_mask)
        hidden_states = attn_outputs + residual

        # Second residual block: MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual
        return hidden_states


class LlamaModel(nn.Module):
    """The full Llama model without any pretraining heads."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.rotary_emb = RotaryPositionEncoding(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-5)

    def reset_parameters(self):
        self.embed_tokens.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, input_ids: Tensor, attn_mask: Tensor) -> Tensor:
        # Convert input token IDs to embeddings
        hidden_states = self.embed_tokens(input_ids)
        # Process through all transformer layers, then the final norm layer
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope=self.rotary_emb, attn_mask=attn_mask)
        hidden_states = self.norm(hidden_states)
        # Return the final hidden states
        return hidden_states


class LlamaForPretraining(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.base_model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def reset_parameters(self):
        self.base_model.reset_parameters()
        self.lm_head.reset_parameters()

    def forward(self, input_ids: Tensor, attn_mask: Tensor) -> Tensor:
        hidden_states = self.base_model(input_ids, attn_mask)
        return self.lm_head(hidden_states)


def create_causal_mask(batch: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """Create a causal mask for self-attention.

    Args:
        batch: Batch of sequences, shape (batch_size, seq_len)
        dtype: Data type of the mask

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    batch_size, seq_len = batch.shape
    mask = torch.full((seq_len, seq_len), float("-inf"), device=batch.device, dtype=dtype) \
                .triu(diagonal=1)
    return mask


def create_padding_mask(batch: Tensor, padding_token_id: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Create a padding mask for a batch of sequences for self-attention.

    Args:
        batch: Batch of sequences, shape (batch_size, seq_len)
        padding_token_id: ID of the padding token
        dtype: Data type of the mask

    Returns:
        Padding mask of shape (batch_size, 1, seq_len, seq_len)
    """
    padded = torch.zeros_like(batch, device=batch.device, dtype=dtype) \
                  .masked_fill(batch == padding_token_id, float("-inf"))
    mask = padded[:,:,None] + padded[:,None,:]
    return mask[:, None, :, :]


# Generator function to create padded sequences of fixed length
class PretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.Dataset, tokenizer: tokenizers.Tokenizer,
                 seq_length: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.bot = tokenizer.token_to_id("[BOT]")
        self.eot = tokenizer.token_to_id("[EOT]")
        self.pad = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Get a sequence of token ids from the dataset. [BOT] and [EOT] tokens
        are added. Clipped and padded to the sequence length.
        """
        seq = self.dataset[index]["text"]
        tokens: list[int] = [self.bot] + self.tokenizer.encode(seq).ids + [self.eot]
        # pad to target sequence length
        toklen = len(tokens)
        if toklen < self.seq_length+1:
            pad_length = self.seq_length+1 - toklen
            tokens += [self.pad] * pad_length
        # return the sequence
        x = torch.tensor(tokens[:self.seq_length], dtype=torch.int64)
        y = torch.tensor(tokens[1:self.seq_length+1], dtype=torch.int64)
        return x, y


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: lr_scheduler.SequentialLR) -> None:
    dist.barrier()
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload),
    )
    load(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id="checkpoint-dist",
    )
    set_state_dict(
        model, optimizer,
        model_state_dict=model_state, optim_state_dict=optimizer_state,
        options=StateDictOptions(broadcast_from_rank0=True, full_state_dict=True, cpu_offload=cpu_offload),
    )
    scheduler.load_state_dict(
        torch.load("checkpoint-dist/lrscheduler.pt", map_location=device),
    )
    dist.barrier()


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: lr_scheduler.SequentialLR) -> None:
    dist.barrier()
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload),
    )
    save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id="checkpoint-dist",
    )
    if dist.get_rank() == 0:
        torch.save(scheduler.state_dict(), "checkpoint-dist/lrscheduler.pt")
    dist.barrier()


# Load the tokenizer and dataset
tokenizer = tokenizers.Tokenizer.from_file("bpe_50K.json")
dataset = datasets.load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train")

# Initialize the distributed environment
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")
rank = dist.get_rank()
world_size = dist.get_world_size()
print(f"World size {world_size}, rank {rank}, local rank {local_rank}. Using {device}")

# Create pretraining model on meta device, on all ranks
with torch.device("meta"):
    model_config = LlamaConfig()
    model = LlamaForPretraining(model_config)

# Convert model from meta device to FSDP2, must shard every component
cpu_offload = False
fsdp_kwargs = {
    # optional: use mixed precision training
    "mp_policy": MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    ),
    # optional: CPU offloading
    "offload_policy": CPUOffloadPolicy() if cpu_offload else None,
    # optional: discard all-gathered parameters after forward pass even on root modules
    # "reshard_after_forward": True,
}
for layer in model.base_model.layers:
    fully_shard(layer, **fsdp_kwargs)
fully_shard(model.base_model, **fsdp_kwargs)
fully_shard(model, **fsdp_kwargs)
model.to_empty(device="cpu" if cpu_offload else device)
model.reset_parameters()
assert isinstance(model, FSDPModule), f"Expected FSDPModule, got {type(model)}"

# Set explicit prefetching on models
# more prefetching uses more memory, but allow more overlap of computation and communication
num_prefetch = 1
if num_prefetch > 1:
    modules = list(model.base_model.layers)
    for i, module in enumerate(modules):
        if i == len(modules) - 1:
            break
        module.set_modules_to_forward_prefetch(modules[i+1:i+num_prefetch+1])
    for i, module in enumerate(modules):
        if i == 0:
            continue
        module.set_modules_to_backward_prefetch(modules[max(0, i-num_prefetch):i])

# Optional: Apply gradient checkpointing on a distributed model (all ranks)
#wrap_policy = functools.partial(
#    transformer_auto_wrap_policy,
#    transformer_layer_cls={LlamaDecoderLayer, nn.Embedding},
#)
#apply_activation_checkpointing(
#    model,
#    checkpoint_wrapper_fn=checkpoint_wrapper,
#    auto_wrap_policy=wrap_policy,
#)

# Training parameters
epochs = 3
learning_rate = 1e-3
batch_size = 64 // world_size
seq_length = 512
num_warmup_steps = 1000
PAD_TOKEN_ID = tokenizer.token_to_id("[PAD]")
model.train()

# DataLoader, optimizer, scheduler, and loss function
# Sampler is needed to shard the dataset across world size
dataset = PretrainingDataset(dataset, tokenizer, seq_length)
sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
dataloader = torch.utils.data.DataLoader(
    dataset,
    sampler=sampler,
    batch_size=batch_size,
    pin_memory=True,  # optional
    shuffle=False,
    num_workers=2,
    prefetch_factor=2,
)
num_training_steps = len(dataloader) * epochs

optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1,
)
warmup_scheduler = lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1, end_factor=1.0, total_iters=num_warmup_steps,
)
cosine_scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_training_steps - num_warmup_steps,
    eta_min=0,
)
scheduler = lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[num_warmup_steps],
)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

# Optional: Compile the model and loss function
#model = torch.compile(model)
#loss_fn = torch.compile(loss_fn)

# if checkpoint-dist dir exists, load the checkpoint to model and optimizer
if os.path.exists("checkpoint-dist"):
    load_checkpoint(model, optimizer, scheduler)

# start training
for epoch in range(epochs):
    pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch_id, batch in enumerate(pbar):
        if batch_id % 1000 == 0:
            save_checkpoint(model, optimizer, scheduler)
        # Explicit prefetching before sending any data to model
        model.unshard()
        # Get batched data, move from CPU to GPU
        input_ids, target_ids = batch
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        # create attention mask: causal mask + padding mask
        attn_mask = create_causal_mask(input_ids) + \
                    create_padding_mask(input_ids, PAD_TOKEN_ID)
        # Extract output from model
        logits = model(input_ids, attn_mask)
        # Compute loss: cross-entropy between logits and target, ignoring padding tokens
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        # Backward with loss and gradient clipping by L2 norm to 1.0
        # Optimizer and gradient clipping works on DTensor
        optimizer.zero_grad(set_to_none=False if cpu_offload else True)
        loss.backward()
        # All-reduce fail if using CPU offloading
        if not cpu_offload:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
        pbar.update(1)
    pbar.close()

# Save the model
save_checkpoint(model, optimizer, scheduler)

# Clean up the distributed environment
dist.destroy_process_group()
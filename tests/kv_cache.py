import torch
import torch.nn as nn

class LatentKVCache(nn.Module):
    """
    Nâng cấp từ logic của Raschka: Quản lý vector nén Latent KV.
    """
    def __init__(self, args):
        super().__init__()
        self.latent_dim = args.kv_lora_rank 
        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size
        
        self.register_buffer(
            "k_cache", 
            torch.zeros((self.max_batch_size, self.max_seq_len, self.latent_dim))
        )

    def update(self, bsz, start_pos, seq_len, x_latent):
        self.k_cache[:bsz, start_pos : start_pos + seq_len] = x_latent
        return self.k_cache[:bsz, :start_pos + seq_len]

class StandardKVCache(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.head_dim = args.dim // args.n_heads
        self.n_kv_heads = args.n_heads
        
        self.register_buffer(
            "k_cache", torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        )
        self.register_buffer(
            "v_cache", torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        )

    def update(self, bsz, start_pos, seq_len, k, v):
        self.k_cache[:bsz, start_pos : start_pos + seq_len] = k
        self.v_cache[:bsz, start_pos : start_pos + seq_len] = v
        return (
            self.k_cache[:bsz, :start_pos + seq_len], 
            self.v_cache[:bsz, :start_pos + seq_len]
        )
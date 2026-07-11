import torch
import torch.nn as nn
from typing import Tuple, Optional

# =============================================================================
# ROTARY POSITIONAL EMBEDDINGS (RoPE) - Llama 3 / DeepSeek
# =============================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Tính toán trước các góc xoay số phức cho RoPE.
    dim: head_dim (phải là số chẵn).
    end: max_seq_len (giới hạn context tối đa).
    """
    # Tính tần số cơ bản
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # (end, dim // 2)
    
    # Tạo số phức dạng e^(i*m*theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Áp dụng phép xoay số phức vào Query và Key.
    xq, xk: (Batch, Seq_Len, n_heads, head_dim)
    """
    # Chuyển sang số phức
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape freqs_cis để broadcasting
    ndim = xq_.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)
    
    # Xoay và trả về dạng thực
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

# =============================================================================
# ABSOLUTE POSITIONAL EMBEDDING (APE) - GPT-2 / Baseline
# =============================================================================

class AbsolutePositionalEmbedding(nn.Module):
    """
    Lớp bọc cho Absolute Position truyền thống.
    Sử dụng phép cộng trực tiếp vào Token Embedding.
    """
    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim) # Bảng tra cứu cố định

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        """
        x: Token embeddings (Batch, Seq_Len, Dim)
        """
        seq_len = x.shape[1]
        # Tạo vector vị trí tương ứng với start_pos
        positions = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        pos_embeddings = self.pos_emb(positions) # (Seq_Len, Dim)
        
        # Phép cộng truyền thống: x + p
        return x + pos_embeddings.unsqueeze(0)
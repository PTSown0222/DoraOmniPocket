import torch

# --- Rotary Positional Encoding (RoPE) Helper Module ---
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_head = d_head
        theta = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('theta', theta)
        
        positions = torch.arange(max_seq_len)
        freqs = torch.outer(positions, self.theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)

    ## NEW ##: Added position_offset for cached inference
    def forward(self, x: torch.Tensor, position_offset: int = 0):
        # x: [B, H, S, D_head]
        seq_len = x.shape[2]
        
        # x_complex: [B, H, S, D_head/2]
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
        # Get precomputed frequencies using the offset
        # freqs_cis: [S, D_head/2] -> [1, 1, S, D_head/2]
        freqs_cis_slice = self.freqs_cis[position_offset : position_offset + seq_len]
        freqs_cis = freqs_cis_slice.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation via element-wise complex multiplication
        x_rotated = x_complex * freqs_cis
        
        # Cast back to real and reshape
        x_out = torch.view_as_real(x_rotated).flatten(3)
        return x_out.type_as(x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, dim, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_dim)
        self.up_proj = nn.Linear(dim, intermediate_dim)
        self.down_proj = nn.Linear(intermediate_dim, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swish = self.act(gate)
        output = self.down_proj(swish * up)
        return output

class MoELayer(nn.Module):
    def __init__(self, dim, intermediate_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(dim, intermediate_dim) for _ in range(num_experts)
        ])
        self.router = nn.Linear(dim, num_experts)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Reshape for expert processing, the compute routing probabilities
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        # shape of router_logits: (batch_size * seq_len, num_experts)
        router_logits = self.router(hidden_states_reshaped)

        # Select top-k experts, than softmax output probabilities will sum to 1
        # output shape: (batch_size * seq_len, k)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Allocate output tensor
        output = torch.zeros(batch_size * seq_len, hidden_dim,
                             device=hidden_states.device,
                             dtype=hidden_states.dtype)

        # Process through selected experts
        unique_experts = torch.unique(top_k_indices)
        for i in unique_experts:
            expert_id = int(i)
            # token_mask (boolean tensor) = which token of the input should use this expert
            # token_mask shape: (batch_size * seq_len,)
            mask = (top_k_indices == expert_id)
            token_mask = mask.any(dim=1)
            assert token_mask.any(), f"Expecting some tokens using expert {expert_id}"

            # select tokens, apply the expert, then add to the output
            expert_input = hidden_states_reshaped[token_mask]
            expert_weight = top_k_probs[mask].unsqueeze(-1)       # shape: (N, 1)
            expert_output = self.experts[expert_id](expert_input) # shape: (N, hidden_dim)
            output[token_mask] += expert_output * expert_weight

        # Reshape back to original shape
        output = output.view(batch_size, seq_len, hidden_dim)
        return output

class MoETransformerLayer(nn.Module):
    def __init__(self, dim, intermediate_dim, num_experts, top_k=2, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.moe = MoELayer(dim, intermediate_dim, num_experts, top_k)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    def forward(self, x):
        # Attention sublayer
        input_x = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        input_x = input_x + attn_output

        # MoE sublayer
        x = self.norm2(input_x)
        moe_output = self.moe(x)
        return input_x + moe_output
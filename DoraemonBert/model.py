import config
import torch
import torch.nn as nn
import torch.nn.functional as F

#---- MoEs Layers -----#

# class MoE(nn.Module):
#     """
#     A Mixture of Experts (MoE) layer with top-k gating.

#     This layer routes each input token to a subset of k experts and computes 
#     a weighted sum of their outputs. Includes a load balancing auxiliary loss.

#     Args:
#         hidden_size (int): The dimensionality of the input features.
#         num_experts (int): The total number of experts in the ensemble.
#         top_k (int): Number of experts to activate for each token.
#     """

#     def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
#         super().__init__()
#         self.num_experts = num_experts
#         self.top_k = top_k
        
#         # Router: Determines the importance of each expert for a given token
#         self.gate = nn.Linear(hidden_size, num_experts)
        
#         # Experts: Individual Feed-Forward Networks
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_size, 4 * hidden_size),
#                 nn.GELU(),
#                 nn.Linear(4 * hidden_size, hidden_size)
#             ) for _ in range(num_experts)
#         ])

#     def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Forward pass with expert routing.

#         Returns:
#             output (torch.Tensor): Weighted sum of expert outputs.
#             aux_loss (torch.Tensor): Load balancing loss to prevent expert collapse.
#         """
#         batch_size, seq_len, hidden_size = x.shape
#         x_flat = x.view(-1, hidden_size)  # Flatten (B, S, H) -> (N, H)
        
#         # 1. Router logits and selection
#         gate_logits = self.gate(x_flat)
#         weights, selected_indices = torch.topk(gate_logits, self.top_k, dim=-1)
#         weights = F.softmax(weights, dim=-1) # (N, top_k)
        
#         # 2. Compute auxiliary loss (Load Balancing)
#         # Prevents one expert from handling all tokens
#         routing_probs = F.softmax(gate_logits, dim=-1) # (N, num_experts)
#         expert_counts = routing_probs.sum(dim=0)
#         aux_loss = (expert_counts.mean() * expert_counts.var()).sum() 
        
#         # 3. Aggregate results (Vectorized approach)
#         output = torch.zeros_like(x_flat)
#         for i in range(self.num_experts):
#             # Find tokens where expert i is one of the top-k
#             mask = (selected_indices == i).any(dim=-1)
#             if mask.any():
#                 # Extract weight of expert i for tokens that selected it
#                 # Find column index of expert i in top_k selection
#                 col_idx = (selected_indices == i).nonzero(as_tuple=True)[1]
#                 w = weights[mask, col_idx].unsqueeze(-1)
                
#                 # Apply expert
#                 expert_out = self.experts[i](x_flat[mask])
#                 output[mask] += w * expert_out
                
#         return output.view(batch_size, seq_len, hidden_size), aux_loss

#----- Doraemon Blocks -----------#
class DoraemonBertBlock(nn.Module):
    """One transformer block"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_prob: float,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout = dropout_prob,
            batch_first = True,
        )

        self.attn_norm = nn.RMSNorm(hidden_size)
        self.ff_norm = nn.RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GeLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
    
    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # self-attention with padding mask and post-norm
        attn_output, _ = self.attention(x, x, x, key_padding_mask=pad_mask)
        x = self.attn_norm(x + attn_output)
        # feed-forward with GeLU activation and post-norm
        ff_output = self.feed_forward(x)
        x = self.ff_norm(x + self.dropout(ff_output))
        return x

class BertPooler(nn.Module):
    """Pooler layer for BERT to process the [CLS] token output."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.activation(x)
        return x        

class BertModel(nn.Module):
    """Backbone of BERT model."""
    def __init__(self, config: BertConfig):
        super().__init__()
        # embedding layers
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
                                            padding_idx=config.pad_id)
        self.type_embeddings = nn.Embedding(config.num_types, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.embeddings_norm = nn.RMSNorm(config.hidden_size)
        self.embeddings_dropout = nn.Dropout(config.dropout_prob)
        # transformer blocks
        self.blocks = nn.ModuleList([
            BertBlock(config.hidden_size, config.num_heads, config.dropout_prob)
            for _ in range(config.num_layers)
        ])
        # [CLS] pooler layer
        self.pooler = BertPooler(config.hidden_size)
 
    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, pad_id: int = 0
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # create attention mask for padding tokens
        pad_mask = input_ids == pad_id
        # convert integer tokens to embedding vectors
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        type_embeddings = self.type_embeddings(token_type_ids)
        token_embeddings = self.word_embeddings(input_ids)
        x = token_embeddings + type_embeddings + position_embeddings
        x = self.embeddings_norm(x)
        x = self.embeddings_dropout(x)
        # process the sequence with transformer blocks
        for block in self.blocks:
            x = block(x, pad_mask)
        # pool the hidden state of the `[CLS]` token
        pooled_output = self.pooler(x[:, 0, :])
        return x, pooled_output
 
class BertPretrainingModel(nn.Module):
    def __init__(self, config: DoraemonBertConfig):
        super().__init__()
        self.bert = BertModel(config)
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size),
        )
        self.nsp_head = nn.Linear(config.hidden_size, 2)
 
    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, pad_id: int = 0
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # Process the sequence with the BERT model backbone
        x, pooled_output = self.bert(input_ids, token_type_ids, pad_id)
        # Predict the masked tokens for the MLM task and the classification for the NSP task
        mlm_logits = self.mlm_head(x)
        nsp_logits = self.nsp_head(pooled_output)
        return mlm_logits, nsp_logits

from minigemma3.rope import apply_rope
from minigemma3.normalization import RMSNorm

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) mechanism.

    GQA is an attention mechanism that strikes a balance between standard
    Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
    - In MHA, each query head has its own key (K) and value (V) head.
    - In MQA, all query heads share a single K and V head.
    - In GQA, multiple query heads are grouped together, and each group shares
      a single K and V head. This reduces the number of parameters and the
      computational load for K and V projections compared to MHA, while often
      maintaining better performance than MQA.
    """

    def __init__(
        self,
        d_in: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int = None,
        qk_norm: bool = False,
        query_pre_attn_scalar: float = None,
        dtype: torch.dtype = None,
    ):
        """
        Initializes the Grouped Query Attention module.

        Args:
            d_in (int): The input dimension of the model (embedding dimension).
            num_heads (int): The total number of query heads.
            num_kv_groups (int): The number of groups for key/value heads. `num_heads` must be divisible by this.
            head_dim (int, optional): The dimension of each attention head. If None, it's inferred from `d_in` and `num_heads`.
            qk_norm (bool, optional): If True, applies RMSNorm to queries and keys before attention. Defaults to False.
            query_pre_attn_scalar (float, optional): A custom scaling factor for queries. If None, defaults to `head_dim**-0.5`.
            dtype (torch.dtype, optional): The data type for the layer's weights.
        """
        super().__init__()
        # Ensure that the number of query heads is a multiple of the number of K/V groups.
        assert num_heads % num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups"
        )

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        # Calculate the number of query heads per K/V group.
        self.group_size = num_heads // num_kv_groups

        # Determine the dimension of each head if not explicitly provided.
        if head_dim is None:
            assert d_in % num_heads == 0, (
                "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            )
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        # The total output dimension from all heads combined.
        self.d_out = num_heads * head_dim

        # === Linear projections for query, key, and value ===
        # Query projection: Maps input to the combined dimension of all query heads.
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        # Key projection: Maps input to the combined dimension of all key heads (one per group).
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        # Value projection: Maps input to the combined dimension of all value heads (one per group).
        self.W_value = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )
        # Output projection: Maps the concatenated attention outputs back to the model's input dimension.
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        # Optional RMSNorm for queries (Q) and keys (K). Normalizing them can improve training stability.
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        # Scaling factor for the query before the attention score calculation.
        # This is a standard practice to prevent the dot products from growing too large,
        # which can lead to vanishing gradients in the softmax function.
        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = head_dim**-0.5

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the Grouped Query Attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, d_in).
            mask (torch.Tensor): Attention mask to prevent attending to certain positions.
            cos (torch.Tensor): Pre-computed cosine values for RoPE.
            sin (torch.Tensor): Pre-computed sine values for RoPE.

        Returns:
            torch.Tensor: The output tensor after attention, of shape (batch_size, num_tokens, d_in).
        """
        # Get the batch size and sequence length from the input tensor.
        b, num_tokens, _ = x.shape

        # 1. Apply linear projections to get the queries, keys, and values.
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # 2. Reshape Q, K, V tensors to separate the heads.
        #    The shape becomes (batch_size, num_heads, num_tokens, head_dim).
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(
            1, 2
        )
        values = values.view(
            b, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)

        # 3. Apply optional normalization to queries and keys.
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # 4. Apply Rotary Positional Embeddings (RoPE) to queries and keys.
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # 5. Repeat keys and values to match the number of query heads for GQA.
        #    Each K/V head is shared across `self.group_size` query heads.
        #    `repeat_interleave` duplicates the K/V heads along the head dimension.
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # 6. Scale queries before computing attention scores.
        queries = queries * self.scaling

        # 7. Compute attention scores (dot product between queries and keys).
        #    - Q shape: (b, num_heads, num_tokens, head_dim)
        #    - K.T shape: (b, num_heads, head_dim, num_tokens)
        #    - Result shape: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        # Apply the attention mask (e.g., to prevent attending to future tokens).
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        # Apply softmax to get attention weights.
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 8. Compute the context vector (weighted sum of values).
        #    - weights shape: (b, num_heads, num_tokens, num_tokens)
        #    - V shape: (b, num_heads, num_tokens, head_dim)
        #    - Result shape: (b, num_heads, num_tokens, head_dim)
        context = attn_weights @ values

        # 9. Reshape the context vector back to the original tensor format.
        #    - `transpose(1, 2)`: Swaps heads and tokens dimensions -> (b, num_tokens, num_heads, head_dim)
        #    - `reshape(...)`: Merges the head and head_dim dimensions -> (b, num_tokens, d_out)
        context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)

        # 10. Apply the final output projection.
        return self.out_proj(context)
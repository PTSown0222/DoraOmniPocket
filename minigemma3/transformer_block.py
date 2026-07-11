class TransformerBlock(nn.Module):
    """
    A single transformer block, which contains an attention mechanism and a feed-forward network.
    This block uses a specific pre/post normalization scheme with residual connections.
    """

    def __init__(self, cfg: dict, attn_type: str):
        super().__init__()
        self.attn_type = attn_type

        # Initialize the Grouped Query Attention mechanism.
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            head_dim=cfg["head_dim"],
            qk_norm=cfg["qk_norm"],
            query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
            dtype=cfg["dtype"],
        )

        # Initialize the Feed-forward network.
        self.ff = FeedForward(cfg)

        # === Normalization Layers ===
        # This implementation uses a somewhat unique normalization style where normalization
        # is applied both *before* the main module (pre-norm) and *after* it (post-norm)
        # but before the residual connection is added.
        self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        mask_global: torch.Tensor,
        mask_local: torch.Tensor,
        cos_global: torch.Tensor,
        sin_global: torch.Tensor,
        cos_local: torch.Tensor,
        sin_local: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer Block.
        """
        # --- Attention Sub-block ---
        # 1. Store the input for the residual connection (skip connection).
        shortcut = x
        # 2. Pre-normalize the input before the attention module.
        x = self.input_layernorm(x)

        # 3. Select the appropriate mask and RoPE parameters based on the attention type.
        #    This allows the model to switch between global and local (sliding window) attention.
        if self.attn_type == "sliding_attention":
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:  # "global_attention"
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global

        # 4. Pass the normalized input through the attention module.
        x_attn = self.att(x, attn_mask, cos, sin)
        # 5. Post-normalize the output of the attention module.
        x_attn = self.post_attention_layernorm(x_attn)
        # 6. Add the residual connection.
        x = shortcut + x_attn

        # --- Feed-Forward Sub-block ---
        # 1. Store the output of the attention block for the next residual connection.
        shortcut = x
        # 2. Pre-normalize the input before the FFN module.
        x_ffn = self.pre_feedforward_layernorm(x)
        # 3. Pass the normalized input through the FFN module.
        x_ffn = self.ff(x_ffn)
        # 4. Post-normalize the output of the FFN module.
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        # 5. Add the second residual connection.
        x = shortcut + x_ffn

        return x
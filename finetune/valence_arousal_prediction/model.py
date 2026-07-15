"""
Architechture of model
"""
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1)
        )
    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: [Batch, Seq, Hidden]

        # 1. "raw score" for each token
        w = self.attention(last_hidden_state) # [Batch, Seq, 1]

        # 2. Masking 
        # w.squeeze(-1) -> [Batch, Seq]
        w = w.squeeze(-1).masked_fill(attention_mask == 0, -1e4)

        # 3. Softmax 
        weights = torch.softmax(w, dim=1).unsqueeze(-1) # [Batch, Seq, 1]

        # 4. Weighted Sum
        context_vector = torch.sum(weights * last_hidden_state, dim=1) # [Batch, Hidden]

        return context_vector

class Expert(nn.Module):
    """a lonely expert"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SparseMoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-K Selection
        topk_weights, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        all_expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=1)
        batch_size = x.size(0)
        final_output = torch.zeros(batch_size, 1, device=x.device)

        # Gather Top-K Results
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]
            weight = topk_weights[:, k].unsqueeze(1)
            idx_view = expert_idx.view(-1, 1, 1).expand(-1, 1, all_expert_outputs.size(-1))
            selected_output = all_expert_outputs.gather(1, idx_view).squeeze(1)
            final_output += weight * selected_output

        return final_output


class ValenceAndArousalModel(nn.Module):
    def __init__(self, model_name, num_experts=4, top_k=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size
        fusion_dim = hidden_size + 2

        # 1. ATTENTION POOLING
        self.pooler = AttentionPooling(hidden_size)

        # 2. SPARSE MOE HEADS
        self.valence_moe = SparseMoELayer(fusion_dim, 256, 1, num_experts, top_k)
        self.arousal_moe = SparseMoELayer(fusion_dim, 256, 1, num_experts, top_k)

        self.loss_fct = CCCLoss()

        # Init weights
        self._init_weights(self.pooler)
        self._init_weights(self.valence_moe)
        self._init_weights(self.arousal_moe)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
             if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
             module.bias.data.zero_()
             module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.ModuleList, nn.Sequential)):
             for sub in module: self._init_weights(sub)
        elif isinstance(module, SparseMoELayer):
            self._init_weights(module.gate)
            self._init_weights(module.experts)
        elif isinstance(module, Expert):
            self._init_weights(module.net)
        elif isinstance(module, AttentionPooling):
            self._init_weights(module.attention)

    def forward(self, input_ids, attention_mask, numerical_features, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        text_feature = self.pooler(outputs.last_hidden_state, attention_mask)
        
        combined_features = torch.cat((text_feature, numerical_features), dim=1)

        val_pred = self.valence_moe(combined_features)
        aro_pred = self.arousal_moe(combined_features)
        logits = torch.cat((val_pred, aro_pred), dim=1)

        loss = None
        if labels is not None:
            loss_v = self.loss_fct(val_pred, labels[:, 0])
            loss_a = self.loss_fct(aro_pred, labels[:, 1])
            loss = 0.5 * loss_v + 0.5 * loss_a

        return {"loss": loss, "logits": logits}
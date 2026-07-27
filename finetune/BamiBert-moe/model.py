
#----------- MLP BamiBert -----------
class MLPClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, config):
        super().__init__()
        mid_features = hidden_size // 2
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mid_features),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mid_features, num_classes)
        )
        
    def forward(self, x):
        return self.mlp(x)

class BanhMiBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bamibert = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(0.5)
        
        self.classifier = MLPClassifier(
            hidden_size=self.bamibert.config.hidden_size,
            num_classes=config.num_classes,
            config=config
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bamibert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits


# -------------- BamiBert-Moes--------------------------#
class MoEClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.k = config.selection_k
        self.num_classes = config.num_classes

        # init gate network (Router)
        self.router = nn.Linear(hidden_size, self.num_experts)

        # lists of experts
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, self.num_classes) for _ in range(self.num_experts)
        ])

    def forward(self, x):
        # x shape: (bz, hidden_sz)
        router_logits = self.router(x)
        gates = F.softmax(router_logits, dim = -1)

        # choose topK Expert
        topk_weights, topk_indices = torch.topk(gates, self.k, dim = -1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim = True) + 1e-7)

        # output
        output = torch.zeros(x.size(0), self.experts[0].out_features, device = x.device)

        for i in range(x.size(0)):
            sample_x = x[i].unsqueeze(0)
            sample_indices = topk_indices[i] 
            sample_weights = topk_weights[i]
            
            for idx, weight in zip(sample_indices, sample_weights):
                expert_output = self.experts[idx](sample_x) 
                output[i] += weight * expert_output.squeeze(0)
        
        return output

class MoEBanhMiBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bamibert = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(0.5)

        # init Model
        self.classifier = MoEClassifier(
            hidden_size = self.bamibert.config.hidden_size,
            num_classes = config.num_classes,
            config = config
        )

    def forward(self, input_ids, attention_mask):
        output = self.bamibert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        pooled_output = output.last_hidden_state[:,0,:]
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits
        
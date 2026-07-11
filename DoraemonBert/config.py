from dataclasses import dataclass

@dataclass
class DoraemonBertConfig:
    emb_dim: int = 1024       
    n_layers: int = 12       
    n_heads: int = 8          
    head_dim: int = 64        
    vocab_size: int = 50257   
    max_seq_len: int = 2048
    norm_eps: float = 1e-6
    dropout: float = 0.1      
    qkv_bias: bool = False

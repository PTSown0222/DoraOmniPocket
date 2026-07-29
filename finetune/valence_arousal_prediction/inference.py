"""Inference Pipeline"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel
from sklearn.preprocessing import StandardScaler
import os
import re
from tqdm import tqdm
try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:
    print("Warning: 'safetensors' not installed. If using .safetensors file, please install it.")

# ============================================================
# 1. CONFIGURATION
# ============================================================
class Config:
    base_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    model_output_dir = "./final_subtask2a_model"
    
    weights_file_name = "model.safetensors" 
    
    train_path = "/kaggle/input/semevaldataset/Dataset/train_subtask2a.csv"
    test_path = "/kaggle/input/semevaldataset/Dataset/TEST_RELEASE_5JAN2026/subtask2a_forecasting_user_marker.csv"
    
    output_file = "pred_subtask2a.csv"
    
    window_size = 8
    max_seq_length = 512
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # MoE Config
    num_experts = 4
    top_k = 2

# ============================================================
# 2. MODEL DEFINITION (SPARSE MOE + ATTENTION POOLING)
# ============================================================
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
        w = self.attention(last_hidden_state) 
        w = w.squeeze(-1).masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(w, dim=1).unsqueeze(-1)
        context_vector = torch.sum(weights * last_hidden_state, dim=1)
        return context_vector

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x): return self.net(x)

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

        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]
            weight = topk_weights[:, k].unsqueeze(1)
            idx_view = expert_idx.view(-1, 1, 1).expand(-1, 1, all_expert_outputs.size(-1))
            selected_output = all_expert_outputs.gather(1, idx_view).squeeze(1)
            final_output += weight * selected_output

        return final_output

class Subtask2aModel(nn.Module):
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

    def forward(self, input_ids, attention_mask, numerical_features):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Attention Pooling thay vì Mean Pooling
        text_feature = self.pooler(outputs.last_hidden_state, attention_mask)
        
        combined_features = torch.cat((text_feature, numerical_features), dim=1)

        val_pred = self.valence_moe(combined_features)
        aro_pred = self.arousal_moe(combined_features)
        
        return val_pred, aro_pred

# ============================================================
# 3. DATA PROCESSING
# ============================================================
def fix_spacing(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+'\s+", "'", text)
    text = re.sub(r"\s+\.", ".", text)
    return text.strip()

def process_dataframe_for_inference(df_path, is_train=False):
    print(f"Processing {df_path}...")
    df = pd.read_csv(df_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    has_forecasting_marker = 'is_forecasting_user' in df.columns
    
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    df['text'] = df['text'].apply(fix_spacing)
    
    processed_data = []
    
    for uid, group in df.groupby('user_id'):
        texts = group['text'].values
        curr_v = group['valence'].values
        curr_a = group['arousal'].values
        
        if is_train:
            for i in range(len(texts)):
                processed_data.append({'numerical_features': [curr_v[i], curr_a[i]]})
        
        else:
            target_indices = []
            if has_forecasting_marker:
                is_true = group['is_forecasting_user'].values
                if 'state_change_valence' in group.columns:
                    is_nan = np.isnan(group['state_change_valence'].values)
                    target_indices = np.where(is_true & is_nan)[0].tolist()
                else:
                    true_indices = np.where(is_true)[0]
                    if len(true_indices) > 0:
                        target_indices = [true_indices[-1]]
            else:
                # Fallback
                target_indices = [len(texts) - 1]

            for idx in target_indices:
                window_texts = []
                for k in range(Config.window_size - 1, -1, -1):
                    i = idx - k
                    if i >= 0:
                        window_texts.append(str(texts[i]))
                full_input = " </s> ".join(window_texts)
                
                processed_data.append({
                    'user_id': uid, 
                    'input_text': full_input,
                    'numerical_features': [curr_v[idx], curr_a[idx]]
                })
            
    return pd.DataFrame(processed_data)

class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['input_text'].values
        self.nums = np.array(df['numerical_features'].tolist(), dtype=np.float32)
        self.tokenizer = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=Config.max_seq_length, return_tensors="pt"
        )
        return {
            "input_ids" : enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'numerical_features': torch.tensor(self.nums[idx], dtype=torch.float)
        }

# ============================================================
# 4. MAIN INFERENCE
# ============================================================
def predict():
    print(">>> [1] Fitting Scaler on TRAINING Data...")
    train_proc = process_dataframe_for_inference(Config.train_path, is_train=True)
    scaler = StandardScaler()
    scaler.fit(np.array(train_proc['numerical_features'].tolist())) 

    print(">>> [2] Processing TEST Data...")
    test_proc = process_dataframe_for_inference(Config.test_path, is_train=False)
    test_nums_scaled = scaler.transform(np.array(test_proc['numerical_features'].tolist()))
    test_proc['numerical_features'] = test_nums_scaled.tolist()
    
    print(f"Unique Predictions to make: {len(test_proc)}")

    print(">>> [3] Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(Config.base_model_name)
    
    # Khởi tạo model với kiến trúc MoE
    model = Subtask2aModel(Config.base_model_name, num_experts=Config.num_experts, top_k=Config.top_k)
    
    weights_path = os.path.join(Config.model_output_dir, Config.weights_file_name)
    
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        
        # Tự động chọn cách load
        if weights_path.endswith(".safetensors"):
            state_dict = safe_load_file(weights_path, device="cpu")
        else:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            
        # strict=False để tránh lỗi nếu có sai khác nhỏ về tên layer (thường không sao)
        model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully!")
    else:
        print(f"ERROR: File not found: {weights_path}")
        return

    model.to(Config.device)
    model.eval()

    print(">>> [4] Running Inference...")
    test_ds = InferenceDataset(test_proc, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False, num_workers=2)

    val_preds = []
    aro_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(Config.device)
            attention_mask = batch['attention_mask'].to(Config.device)
            numerical_features = batch['numerical_features'].to(Config.device)

            p_val, p_aro = model(input_ids, attention_mask, numerical_features)
            val_preds.extend(p_val.cpu().numpy().flatten())
            aro_preds.extend(p_aro.cpu().numpy().flatten())

    print(">>> [5] Creating Submission File...")
    submission = pd.DataFrame({
        'user_id': test_proc['user_id'],
        'pred_state_change_valence': val_preds,
        'pred_state_change_arousal': aro_preds
    })
    
    submission.to_csv(Config.output_file, index=False)
    print(f"Done! Submission saved to: {Config.output_file}")
    print(submission.head())

if __name__ == "__main__":
    predict()
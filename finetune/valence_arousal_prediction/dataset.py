import torch
import torch.nn as nn

def fix_spacing(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+'\s+", "'", text)
    text = re.sub(r"\s+\.", ".", text)
    return text.strip()

def prepare_data(path):
    print(f">>> Loading and Processing Data from {path}...")
    df = pd.read_csv(path)

    # 1. Sắp xếp chuẩn Time-Series
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    df['text'] = df['text'].apply(fix_spacing)

    new_data = []

    for uid, group in df.groupby('user_id'):
        texts = group['text'].values

        curr_v = group['valence'].values
        curr_a = group['arousal'].values
        #time_logs = group['time_log'].values
        delta_v = group['state_change_valence'].values
        delta_a = group['state_change_arousal'].values

        original_indices = group.index.tolist()

        for i in range(len(texts)):
            # check None thì bỏ qua
            if np.isnan(delta_v[i]) or np.isnan(delta_a[i]):
                continue

            # Creating Sliding Window Text: [Text_i-2, Text_i-1, Text_i]
            window_texts = []
            for k in range(Config.window_size - 1, -1, -1): # k=2, 1, 0
                if i - k >= 0:
                    window_texts.append(str(texts[i-k]))

            # Nối bằng token ngăn cách </s> của RoBERTa
            full_input = " </s> ".join(window_texts)

            new_data.append({
                'user_id': uid,
                'input_text': full_input,
                # Nếu xài thời gian thì thêm time_logs[i] vào
                'numerical_features': [curr_v[i], curr_a[i]],
                'labels': [delta_v[i], delta_a[i]]
            })

        df_new = pd.DataFrame(new_data)
        scaler = StandardScaler()
        nums_matrix = np.array(df_new['numerical_features'].tolist())
        nums_scaled = scaler.fit_transform(nums_matrix)
        df_new['numerical_features'] = nums_scaled.tolist()
    
    return df_new

class EmotionDatasetSubtask2a(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['input_text'].values
        self.nums = np.array(df['numerical_features'].tolist(), dtype=np.float32)
        self.labels = np.array(df['labels'].tolist(), dtype=np.float32)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=Config.max_seq_length,
            return_tensors="pt"
        )

        return {
            "input_ids" : enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'numerical_features': torch.tensor(self.nums[idx], dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
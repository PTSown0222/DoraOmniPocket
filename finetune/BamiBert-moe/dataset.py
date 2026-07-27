

# Preprocessing
def preprocess_data(
    df,
    text_col,
    label_col,
    tokenizer,
    segmenter,
    max_len,
):
    vnp = VnCoreNLP(config.vncorenlp_path,annotators="wseg")

    # cleaned text
    cleaned_texts = [clean_text(str(t)) for t in df[text_col].tolist()]
    
    # save cleaned text and segmenter
    segmented_texts = []
    for text in cleaned_texts:
        word_segment = segmenter.tokenize(text)
        segmented_texts.append(" ".join(w for sentence in word_segment for w in sentence))

    encoded_texts = tokenizer(
        text = segmented_texts,
        padding = "max_length",
        truncation = True,
        max_length = max_len,
        return_tensors = "pt",
    )

    label_tensor = torch.tensor(df[label_col].tolist(), dtype=torch.long)
    
    return {
        "input_ids": encoded_texts['input_ids'],
        "attention_mask": encoded_texts['attention_mask'],
        "labels": label_tensor,
    }

# Custom Dataset
class SentimentDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
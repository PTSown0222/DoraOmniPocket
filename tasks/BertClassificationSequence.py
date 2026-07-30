import dataclasses
import functools

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
from torch import Tensor

# BERT config and model defined previously
@dataclasses.dataclass
class BertConfig:
    """Configuration for BERT model."""
    vocab_size: int = 30522
    num_layers: int = 12
    hidden_size: int = 768
    num_heads: int = 12
    dropout_prob: float = 0.1
    pad_id: int = 0
    max_seq_len: int = 512
    num_types: int = 2

class BertBlock(nn.Module):
    """One transformer block in BERT."""
    def __init__(self, hidden_size: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads,
                                               dropout=dropout_prob, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
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
        self.embeddings_norm = nn.LayerNorm(config.hidden_size)
        self.embeddings_dropout = nn.Dropout(config.dropout_prob)
        # transformer blocks
        self.blocks = nn.ModuleList([
            BertBlock(config.hidden_size, config.num_heads, config.dropout_prob)
            for _ in range(config.num_layers)
        ])
        # [CLS] pooler layer
        self.pooler = BertPooler(config.hidden_size)

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, pad_id: int = 0,
                ) -> tuple[Tensor, Tensor]:
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

# Define new BERT model for sequence classification
class BertForSequenceClassification(nn.Module):
    """BERT model for GLUE tasks."""
    def __init__(self, config: BertConfig, num_labels: int):
        super().__init__()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids: Tensor, pad_id: int = 0) -> Tensor:
        # pooled_output corresponds to the [CLS] token
        token_type_ids = torch.zeros_like(input_ids)
        seq_output, pooled_output = self.bert(input_ids, token_type_ids, pad_id=pad_id)
        logits = self.classifier(pooled_output)
        return logits

# Load GLUE dataset (e.g., 'sst2' for sentiment classification)
task = "sst2"
dataset = load_dataset("glue", task)
num_labels = 2  # dataset["train"]["label"] is either 0 or 1

# Load the pretrained BERT tokenizer
TOKENIZER_PATH = "wikitext-2_wordpiece.json"
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# Setup dataloader for training and validation datasets
def collate(batch: list[dict], tokenizer: Tokenizer, max_len: int) -> tuple[Tensor, Tensor]:
    """Collate variable-length sequences in the dataset."""
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    pad_id = tokenizer.token_to_id("[PAD]")
    sentences: list[str] = [item["sentence"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    input_ids = []
    for sentence in sentences:
        seq = [cls_id]
        seq.extend(tokenizer.encode(sentence).ids)
        if len(seq) >= max_len:
            seq = seq[:max_len-1]
        seq.append(sep_id)
        num_pad = max_len - len(seq)
        seq.extend([pad_id] * num_pad)
        input_ids.append(seq)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    return input_ids, labels

batch_size = 16
max_len = 128
collate_fn = functools.partial(collate, tokenizer=tokenizer, max_len=max_len)
train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size,
                                           shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(dataset["validation"], batch_size=batch_size,
                                         shuffle=False, collate_fn=collate_fn)

# Create classification model with a pretrained foundation BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = BertConfig()
model = BertForSequenceClassification(config, num_labels)
model.to(device)
model.bert.load_state_dict(torch.load("bert_model.pth", map_location=device))

# Training setup
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    # Training
    with tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for batch in pbar:
            # get batched data
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            # forward pass
            logits = model(input_ids, torch.zeros_like(input_ids))
            # backward pass
            optimizer.zero_grad()
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            # update progress bar
            pbar.set_postfix(loss=float(loss))
            pbar.update(1)

    # Validation: Keep track of the average loss and accuracy
    model.eval()
    val_loss, num_matches, num_batches, num_samples = 0, 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            # get batched data
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            # forward pass on validation data
            logits = model(input_ids)
            # compute loss
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
            num_batches += 1
            # compute accuracy
            predictions = logits.argmax(dim=-1)
            num_matches += (predictions == labels).sum().item()
            num_samples += len(labels)
    avg_loss = val_loss / num_batches
    acc = num_matches / num_samples
    print(f"Validation {epoch+1}/{num_epochs}: acc {acc:.4f}, avg loss {avg_loss:.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), f"bert_model_glue_sst2.pth")
import collections
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

# Define new BERT model for question answering
class BertForQuestionAnswering(nn.Module):
    """BERT model for SQuAD question answering."""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel(config)
        # Two outputs: start and end position logits
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        pad_id: int = 0,
    ) -> tuple[Tensor, Tensor]:
        # Get sequence output from BERT (batch_size, seq_len, hidden_size)
        seq_output, pooled_output = self.bert(input_ids, token_type_ids, pad_id=pad_id)
        # Project to start and end logits
        logits = self.qa_outputs(seq_output)  # (batch_size, seq_len, 2)
        start_logits = logits[:, :, 0]  # (batch_size, seq_len)
        end_logits = logits[:, :, 1]    # (batch_size, seq_len)
        return start_logits, end_logits

# Load SQuAD dataset for question answering
dataset = load_dataset("squad")

# Load the pretrained BERT tokenizer
TOKENIZER_PATH = "wikitext-2_wordpiece.json"
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# Setup collate function to tokenize question-context pairs for the model
def collate(batch: list[dict], tokenizer: Tokenizer, max_len: int,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collate question-context pairs for the model."""
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    pad_id = tokenizer.token_to_id("[PAD]")

    input_ids_list = []
    token_type_ids_list = []
    start_positions = []
    end_positions = []

    for item in batch:
        # Tokenize question and context
        question, context = item["question"], item["context"]
        question_ids = tokenizer.encode(question).ids
        context_ids = tokenizer.encode(context).ids

        # Build input: [CLS] question [SEP] context [SEP]
        input_ids = [cls_id, *question_ids, sep_id, *context_ids, sep_id]
        token_type_ids = [0] * (len(question_ids)+2) + [1] * (len(context_ids)+1)

        # Truncate or pad to max length
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            token_type_ids = token_type_ids[:max_len]
        else:
            input_ids.extend([pad_id] * (max_len - len(input_ids)))
            token_type_ids.extend([1] * (max_len - len(token_type_ids)))

        # Find answer position in tokens: Answer may not be in the context
        start_pos = end_pos = 0
        if len(item["answers"]["text"]) > 0:
            answers = tokenizer.encode(item["answers"]["text"][0]).ids
            # find the context offset of the answer in context_ids
            for i in range(len(context_ids) - len(answers) + 1):
                if context_ids[i:i+len(answers)] == answers:
                    start_pos = i + len(question_ids) + 2
                    end_pos = start_pos + len(answers) - 1
                    break
            if end_pos >= max_len:
                start_pos = end_pos = 0  # answer is clipped, hence no answer

        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        start_positions.append(start_pos)
        end_positions.append(end_pos)

    input_ids_list = torch.tensor(input_ids_list)
    token_type_ids_list = torch.tensor(token_type_ids_list)
    start_positions = torch.tensor(start_positions)
    end_positions = torch.tensor(end_positions)
    return (input_ids_list, token_type_ids_list, start_positions, end_positions)

batch_size = 16
max_len = 384  # Longer for Q&A to accommodate context
collate_fn = functools.partial(collate, tokenizer=tokenizer, max_len=max_len)
train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size,
                                           shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(dataset["validation"], batch_size=batch_size,
                                         shuffle=False, collate_fn=collate_fn)

# Create Q&A model with a pretrained foundation BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = BertConfig()
model = BertForQuestionAnswering(config)
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
            input_ids, token_type_ids, start_positions, end_positions = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)
            # forward pass
            start_logits, end_logits = model(input_ids, token_type_ids)
            # backward pass
            optimizer.zero_grad()
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = start_loss + end_loss
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
            input_ids, token_type_ids, start_positions, end_positions = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)
            # forward pass on validation data
            start_logits, end_logits = model(input_ids, token_type_ids)
            # compute loss
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = start_loss + end_loss
            val_loss += loss.item()
            num_batches += 1
            # compute accuracy
            pred_start = start_logits.argmax(dim=-1)
            pred_end = end_logits.argmax(dim=-1)
            match = (pred_start == start_positions) & (pred_end == end_positions)
            num_matches += match.sum().item()
            num_samples += len(start_positions)

    avg_loss = val_loss / num_batches
    acc = num_matches / num_samples
    print(f"Validation {epoch+1}/{num_epochs}: acc {acc:.4f}, avg loss {avg_loss:.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), f"bert_model_squad.pth")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn.utils import clip_grad_norm_

# Training parameters
epochs = 10
learning_rate = 1e-4
batch_size = 32

# Define learning rate schedulers
warmup_steps = 10
total_steps = 100
min_lr = 1e-4

# train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertPretrainingModel(DoraemonBertConfig()).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

warmup_lr = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
cosine_lr = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)
combined_lr = SequentialLR(optimizer, schedulers=[warmup_lr, cosine_lr], milestones=[warmup_steps])

loss_fn = nn.CrossEntropyLoss()
 
for epoch in range(epochs):
    pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in pbar:
        # get batched data
        input_ids, token_type_ids, is_random_next, masked_pos, masked_labels = batch
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        is_random_next = is_random_next.to(device)
        masked_labels = masked_labels.to(device)
        # extract output from model
        mlm_logits, nsp_logits = model(input_ids, token_type_ids)
        # MLM loss: masked_positions is a list of tuples of (B, S), extract the
        # corresponding logits from tensor mlm_logits of shape (B, S, V)
        batch_indices, token_positions = zip(*masked_pos)
        mlm_logits = mlm_logits[batch_indices, token_positions]
        mlm_loss = loss_fn(mlm_logits, masked_labels)
        # Compute the loss for the NSP task
        nsp_loss = loss_fn(nsp_logits, is_random_next)
        # backward with total loss
        total_loss = mlm_loss + nsp_loss
        pbar.set_postfix(MLM=mlm_loss.item(), NSP=nsp_loss.item(), Total=total_loss.item())
        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        combined_lr.step()
        scheduler.step()
        pbar.update(1)
    pbar.close()
 
# Save the model
torch.save(model.state_dict(), "bert_pretraining_model.pth")
torch.save(model.bert.state_dict(), "bert_model.pth")
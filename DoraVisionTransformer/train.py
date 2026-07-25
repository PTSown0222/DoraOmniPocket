from DoraVisionTransformer.model import DoraVisionLanguageModel
import torch.optim as optim

EPOCHS = 5

optimizer = optim.AdamW(vlm.parameters(), lr=1e-5)
for epoch in range(EPOCHS):
    for X, y in dataloader:
        pass
        
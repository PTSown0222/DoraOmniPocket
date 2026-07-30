import torch
import torch.nn as nn

class CCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        mu_pred = torch.mean(y_pred)
        mu_true = torch.mean(y_true)

        var_pred = torch.var(y_pred, unbiased=False)
        var_true = torch.var(y_true, unbiased=False)

        covariance = torch.mean((y_pred - mu_pred) * (y_true - mu_true))

        numerator = 2 * covariance
        denominator = var_pred + var_true + (mu_pred - mu_true)**2 + self.eps

        return 1.0 - (numerator / denominator)
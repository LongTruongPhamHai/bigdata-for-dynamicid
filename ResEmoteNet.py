import torch
import torch.nn as nn


class ResEmoteNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        # trả embedding zero
        return torch.zeros((x.shape[0], 512), device=x.device)

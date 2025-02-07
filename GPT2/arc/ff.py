import torch
from torch import nn


class GELU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=cfg["emb_dim"], out_features=4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(in_features=4 * cfg["emb_dim"], out_features=cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

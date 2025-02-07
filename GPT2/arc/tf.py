import torch
from torch import nn
from LLM_creation.arc.attention import MultiHeadAttention_v3
from LLM_creation.arc.ff import FeedForward
from LLM_creation.arc.norm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention_v3(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg=cfg)
        self.norm1 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.norm2 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(p=cfg["drop_rate"])

    def forward(self, x):

        shortcut = x  # residual connection for attention
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x += shortcut

        shortcut = x  # residual connection for ff bloc
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x += shortcut

        return x

import torch
from torch import nn
from LLM_creation.arc.tf import TransformerBlock
from LLM_creation.arc.norm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(num_embeddings=cfg["vocab_size"], embedding_dim=cfg["emb_dim"])
        self.pos_emb = nn.Embedding(num_embeddings=cfg["context_length"], embedding_dim=cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])


        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg=cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(emb_dim=cfg["emb_dim"])
        self.out_head = nn.Linear(
            in_features=cfg["emb_dim"], out_features=cfg["vocab_size"], bias=False
        )


    
    def forward(self, in_idx: torch.Tensor):
        batch_size, seq_len = in_idx.shape
        in_idx = in_idx.type(dtype=torch.long)
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
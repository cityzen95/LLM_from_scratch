import torch
from arc.model import GPTModel_v2 # customized implementation with RoPE, weight tying, combined QKV projections, KV cache

GPT_CONFIG_124M = {
 "vocab_size": 50257, # Vocabulary size
 "context_length": 1024, # Context length
 "emb_dim": 768, # Embedding dimension
 "n_heads": 12, # Number of attention heads
 "n_layers": 12, # Number of layers
 "drop_rate": 0.1, # Dropout rate
 "qkv_bias": False # Query-Key-Value bias
}

if __name__ == "__main__":
    sequence = torch.rand(12).unsqueeze(dim=0)
    model = GPTModel_v2(cfg=GPT_CONFIG_124M)
    print(f"Total number trainable params: {sum(i.numel() for i in model.parameters()):,}")

    # Training (full sequence, no cache)
    model.train()
    logits = model(sequence)
    print(logits.shape)

    # Inference (autoregressive with KV cache)
    with torch.inference_mode():
        model.eval()
        logits, cache = model(sequence, use_cache=True)
        print(logits.shape)
        print(len(cache))
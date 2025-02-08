# LLM_from_scratch
- Building LLMs from scratch following the Build a Large Language Model (From Scratch) book from S. Raschka  
- Deviated from the book and created a customized version of the GPT2 architecture:
  - Replaced absolute positional embeddings with RoPE in attention.
  - An additional implementation of MultiHeadAttention was added using the efficient PyTorch F.scaled_dot_product_attention.
  - Combined QKV projections into a single matrix multiplication. (reduces the number of separate matrix multiplications)
  - Added weight tying between input and output embeddings
  - Added support for the KV cache mechanism.

The project was implemented for educational purposes.

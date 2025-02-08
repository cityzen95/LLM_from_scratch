import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Any, Optional, List


class CausalAttention(nn.Module):
    def __init__(self, d_in, context_len, d_out, droupout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(p=droupout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_weight = attn_weights @ values
        return context_weight


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, droupout, num_heads, qkv_bias=False
    ):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                CausalAttention(
                    d_in=d_in,
                    context_len=context_length,
                    d_out=d_out,
                    droupout=droupout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert not (d_out % num_heads), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // self.num_heads

        self.W_query = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_key = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_value = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )  # causal mask

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # adding num_heads dimension for efficient multi head mm
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # transposing from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec)
        return context_vec


# Implementation using PyTorch F.scaled_dot_product_attention
class MultiHeadAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Combined QKV projection:
        self.W_qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(p=dropout)
        self.context_length = context_length

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function that rotates half the dimensions of the tensor."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embeddings to q and k.
        
        Args:
            q, k: tensors of shape [batch, num_heads, num_tokens, head_dim]
            pos_ids: tensor of shape [num_tokens] with positional indices
        Returns:
            q_rot, k_rot: tensors with RoPE applied.
        """
        batch_size, num_heads, num_tokens, head_dim = q.shape
        device = q.device

        # Compute theta as a vector of shape [head_dim//2]
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        # Compute angles: shape [num_tokens, head_dim//2]
        angles = pos_ids.unsqueeze(-1).float() * theta  # [num_tokens, head_dim//2]
        
        # Compute cosine and sine: shape [1, 1, num_tokens, head_dim//2]
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        # Expand last dimension to match head_dim (which should be even)
        cos = cos.repeat(1, 1, 1, 2)  # [1, 1, num_tokens, head_dim]
        sin = sin.repeat(1, 1, 1, 2)  # same shape
        
        # Apply RoPE to q and k
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot.to(q.dtype), k_rot.to(k.dtype)

    def forward(self, x: torch.Tensor, 
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False, pos_offset: int = 0):
        """
        Args:
            x: input tensor [batch, seq, dim]
            past_kv: tuple of (past_keys, past_values) each of shape [batch, num_heads, seq, head_dim]
            use_cache: whether to return updated KV cache
            pos_offset: positional offset for RoPE calculations
        """
        b, num_tokens, d_in = x.shape

        # Combined QKV projection and split
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to [batch, num_tokens, num_heads, head_dim] and transpose to [batch, num_heads, num_tokens, head_dim]
        q = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache: concatenate past keys and values if provided.
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Calculate the correct starting position and current positions.
        if past_kv is None:
            start_pos = 0
        else:
            start_pos = past_kv[0].shape[2]  # previous sequence length
        # Incorporate pos_offset if provided:
        current_pos = start_pos + pos_offset + torch.arange(num_tokens, device=x.device)

        # Apply RoPE to queries and keys
        q, k = self.apply_rope(q, k, current_pos)

        # Compute attention using PyTorch's scaled_dot_product_attention
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=(past_kv is None),  # Use full causal mask only on first step
                dropout_p=self.dropout.p
            )

        
        # Reformat the output to [batch, seq, d_out]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(b, num_tokens, self.d_out)
        output = self.out_proj(attn_output)

        # Return updated cache if requested
        if use_cache:
            return output, (k, v)
        return output, None


# manual implementation of SDPA
class MultiHeadAttention_v2M(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Combined QKV projection:
        self.W_qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(p=dropout)
        self.context_length = context_length

        # Create a causal mask buffer of shape [context_length, context_length]
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function that rotates half the dimensions of the tensor."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies Rotary Positional Embeddings (RoPE) to queries and keys.
        Args:
            q, k: Tensors of shape [batch, num_heads, num_tokens, head_dim]
            pos_ids: Tensor of shape [num_tokens] containing the positions.
        Returns:
            Tuple (q_rot, k_rot) with RoPE applied.
        """
        batch_size, num_heads, num_tokens, head_dim = q.shape
        device = q.device

        # Compute theta: shape [head_dim//2]
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        # Compute angles: shape [num_tokens, head_dim//2]
        angles = pos_ids.unsqueeze(-1).float() * theta  
        # Compute cosine and sine with explicit unsqueezing to match [1, 1, num_tokens, head_dim//2]
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        # Expand to full head_dim by repeating along the last dimension (assuming head_dim is even)
        cos = cos.repeat(1, 1, 1, 2)  # resulting shape: [1, 1, num_tokens, head_dim]
        sin = sin.repeat(1, 1, 1, 2)  # same shape

        # Apply RoPE: perform element-wise rotation using the computed cosine and sine.
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot.to(q.dtype), k_rot.to(k.dtype)

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False, pos_offset: int = 0):
        """
        Args:
            x: input tensor of shape [batch, seq, d_in]
            past_kv: optional tuple (past_keys, past_values) each of shape [batch, num_heads, seq, head_dim]
            use_cache: whether to return updated key/value cache
            pos_offset: positional offset for RoPE calculations
        Returns:
            output: tensor of shape [batch, seq, d_out]
            (optionally) a tuple (k, v) for the updated cache if use_cache is True.
        """
        b, num_tokens, d_in = x.shape

        # Compute combined QKV projection and split into queries, keys, and values.
        qkv = self.W_qkv(x)  # shape: [b, num_tokens, 3 * d_out]
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape and transpose: final shape [b, num_heads, num_tokens, head_dim]
        q = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle key/value caching: if past_kv is provided, append current k and v along the sequence dimension.
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Determine starting position for positional embeddings.
        start_pos = past_kv[0].shape[2] if past_kv is not None else 0
        # Incorporate any additional offset and build a positions tensor.
        current_pos = start_pos + pos_offset + torch.arange(num_tokens, device=x.device)

        # Apply RoPE to queries and keys.
        q, k = self.apply_rope(q, k, current_pos)

        # --- Manual Scaled Dot-Product Attention ---
        scale = self.head_dim ** 0.5
        # Compute raw attention scores [b, num_heads, num_tokens, seq_total]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Apply causal mask:
        # Ensure that only the first "num_tokens" positions are used from the mask.
        mask = self.mask[:attn_scores.size(-2), :attn_scores.size(-1)]
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        # Softmax over the last dimension (keys dimension) and apply dropout.
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute attention output by weighting the values.
        attn_output = torch.matmul(attn_weights, v)

        # Rearrange the output to [b, num_tokens, d_out]
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        output = self.out_proj(attn_output)

        # Return updated key/value cache if requested.
        if use_cache:
            return output, (k, v)
        return output, None

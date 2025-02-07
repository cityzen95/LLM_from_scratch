import torch
from torch import nn
from LLM_creation.arc.rope import RotaryPositionalEmbeddings


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


class MultiHeadAttention_v3(nn.Module):
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
        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
        )
        self.context_length = context_length

    def forward(self, x):
        b, num_tokens, _ = x.shape
        # Combined projection:
        qkv = self.W_qkv(x)  # Shape: (b, num_tokens, 3 * d_out)
        # Split into queries, keys, values:
        qkv = qkv.reshape(b, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # Now shape: (3, b, num_tokens, n_h, h_d)
        queries, keys, values = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # Each: (b, num_tokens, n_h, h_d)

        # Apply RoPE while tensor is in shape [b, seq_len, n_h, h_d]
        queries = self.rope(queries)
        keys = self.rope(keys)

        # Now transpose to shape [b, n_h, seq_len, h_d] for dot product:
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scale = self.head_dim**0.5
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale

        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens], float("-inf")
        )
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, values)
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        output = self.out_proj(context)
        return output

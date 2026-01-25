import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSelfAttention(nn.Module):
    """
    Linear attention using feature map phi(x) = elu(x) + 1
    Reference: "Transformers are RNNs" / Performer-style approximation
    """

    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        B, L, D = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = self._feature_map(q)
        k = self._feature_map(k)

        # Causal prefix sums
        kv = torch.einsum("bhld,bhlm->bhmd", k, v)
        k_sum = k.sum(dim=2, keepdim=True)

        z = 1 / torch.einsum("bhld,bhmd->bhlm", q, k_sum + 1e-6)
        out = torch.einsum("bhld,bhmd->bhlm", q, kv)
        out = out * z

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out)


class LinearSASRecBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()

        self.attn = LinearSelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class LinearSASRec(nn.Module):
    def __init__(
        self,
        num_items,
        hidden_size,
        num_layers,
        num_heads,
        dropout,
        max_seq_len,
    ):
        super().__init__()

        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList(
            [
                LinearSASRecBlock(hidden_size, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, num_items + 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, seqs):
        B, L = seqs.size()

        pos = torch.arange(L, device=seqs.device).unsqueeze(0).expand(B, L)

        x = self.item_emb(seqs) + self.pos_emb(pos)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Use last position
        logits = self.out(x[:, -1, :])
        return logits

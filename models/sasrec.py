import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, num_items, hidden_size, num_layers, num_heads, dropout):
        super().__init__()

        self.num_items = num_items
        self.hidden_size = hidden_size

        # Item embedding (padding index = 0)
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=hidden_size,
            padding_idx=0
        )

        # Positional embedding
        self.position_embedding = nn.Embedding(
            num_embeddings=1000,  # max sequence length (safe upper bound)
            embedding_dim=hidden_size
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Final layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_sequences):
        """
        Args:
            input_sequences: Tensor of shape [batch_size, seq_len]
        Returns:
            logits: Tensor of shape [batch_size, num_items]
        """
        device = input_sequences.device
        batch_size, seq_len = input_sequences.size()

        # Item embeddings
        item_emb = self.item_embedding(input_sequences)

        # Positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        x = item_emb + pos_emb
        x = self.dropout(x)

        # Causal mask (prevent attending to future)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()

        # Transformer encoder
        x = self.encoder(x, mask=causal_mask)
        x = self.layer_norm(x)

        # Use the last position
        final_state = x[:, -1, :]  # [batch_size, hidden_size]

        # Compute logits against all items
        logits = torch.matmul(
            final_state,
            self.item_embedding.weight.transpose(0, 1)
        )

        return logits

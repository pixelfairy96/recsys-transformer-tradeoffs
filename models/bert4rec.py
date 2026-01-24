import torch
import torch.nn as nn


class BERT4Rec(nn.Module):
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

        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Item embedding (+1 for padding index 0)
        self.item_embedding = nn.Embedding(
            num_items + 1, hidden_size, padding_idx=0
        )

        # Positional embedding
        self.position_embedding = nn.Embedding(
            max_seq_len, hidden_size
        )

        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        # Output layer: predict items
        self.output = nn.Linear(hidden_size, num_items + 1)

    def forward(self, input_seq):
        """
        input_seq: (B, L)
        returns logits: (B, num_items)
        """

        device = input_seq.device
        batch_size, seq_len = input_seq.size()

        # Positions
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        # Embeddings
        item_emb = self.item_embedding(input_seq)
        pos_emb = self.position_embedding(positions)

        x = item_emb + pos_emb
        x = self.emb_dropout(x)

        # Padding mask (True where padding)
        padding_mask = (input_seq == 0)

        # Bidirectional self-attention
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        x = self.layer_norm(x)

        # Take representation of last non-padding position
        lengths = (input_seq != 0).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)

        last_hidden = x[torch.arange(batch_size, device=device), lengths]

        logits = self.output(last_hidden)

        return logits

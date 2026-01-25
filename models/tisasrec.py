import torch
import torch.nn as nn


class TiSASRec(nn.Module):
    def __init__(
        self,
        num_items,
        num_time_bins,
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

        # Time-interval embedding (+1 for padding)
        self.time_embedding = nn.Embedding(
            num_time_bins + 1, hidden_size, padding_idx=0
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

        self.output = nn.Linear(hidden_size, num_items + 1)

    def forward(self, input_seq, time_seq):
        """
        input_seq: (B, L)
        time_seq:  (B, L)
        returns logits: (B, num_items)
        """

        device = input_seq.device
        batch_size, seq_len = input_seq.size()

        # Positions
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        # Embeddings
        item_emb = self.item_embedding(input_seq)
        pos_emb = self.position_embedding(positions)
        time_emb = self.time_embedding(time_seq)

        x = item_emb + pos_emb + time_emb
        x = self.emb_dropout(x)

        # Padding mask
        padding_mask = (input_seq == 0)

        # Bidirectional attention (same as SASRec, but richer embeddings)
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        x = self.layer_norm(x)

        # Last non-padding position
        lengths = (input_seq != 0).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)

        last_hidden = x[torch.arange(batch_size, device=device), lengths]

        logits = self.output(last_hidden)

        return logits
 
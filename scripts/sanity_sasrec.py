import torch
from models.sasrec import SASRec

batch_size = 2
seq_len = 50
num_items = 4000

model = SASRec(
    num_items=num_items,
    hidden_size=64,
    num_layers=2,
    num_heads=2,
    dropout=0.2
)

dummy_input = torch.randint(
    low=1,
    high=num_items,
    size=(batch_size, seq_len)
)

logits = model(dummy_input)

print("Logits shape:", logits.shape)

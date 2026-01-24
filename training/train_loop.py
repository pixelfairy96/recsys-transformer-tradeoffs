import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    sequences, targets = zip(*batch)

    sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    targets = torch.tensor(targets, dtype=torch.long)

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )

    return padded_sequences, targets


def train_one_epoch(model, train_data, optimizer, device, batch_size):
    model.train()
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch
    )

    total_loss = 0.0

    torch.cuda.synchronize()
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(sequences)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    torch.cuda.synchronize()
    epoch_time = time.time() - start_time

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    return total_loss / len(loader), epoch_time, peak_memory


@torch.no_grad()
def evaluate(model, eval_data, device, batch_size):
    model.eval()

    loader = DataLoader(
    eval_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch
    )

    correct = 0
    total = 0

    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        logits = model(sequences)
        preds = logits.argmax(dim=1)

        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return correct / total

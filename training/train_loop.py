import time
import torch
import torch.nn as nn
from training.metrics import evaluate_ndcg


def train_one_epoch(model, data, optimizer, device, batch_size):
    model.train()
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    total_loss = 0.0

    for batch in _batch_iter(data, batch_size):
        optimizer.zero_grad()

        # -----------------------------
        # Unpack batch
        # -----------------------------
        if model.__class__.__name__ == "TiSASRec":
            # (seqs, times, targets)
            seqs, times, targets = batch
            seqs = seqs.to(device)
            times = times.to(device)
            targets = targets.to(device)
            logits = model(seqs, times)
        else:
            # (seqs, targets) or (seqs, _, targets)
            if len(batch) == 3:
                seqs, _, targets = batch
            else:
                seqs, targets = batch

            seqs = seqs.to(device)
            targets = targets.to(device)
            logits = model(seqs)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_mem = 0.0

    epoch_time = time.time() - start_time

    return total_loss / len(data), epoch_time, peak_mem


@torch.no_grad()
def evaluate(model, data, device, batch_size, k=10):
    """
    Proper ranking evaluation.
    Returns NDCG@k.
    """
    model.eval()
    return evaluate_ndcg(model, data, device, batch_size, k=k)


def _batch_iter(data, batch_size):
    """
    data elements:
      - SASRec / BERT4Rec:
          (seq_list, target)
      - TiSASRec:
          (seq_list, time_list, target)

    Converts lists -> tensors.
    """
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        cols = list(zip(*batch))

        tensors = []
        for col in cols:
            tensors.append(torch.tensor(col, dtype=torch.long))

        yield tuple(tensors)

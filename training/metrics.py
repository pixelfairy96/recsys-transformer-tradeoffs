import torch
import numpy as np


@torch.no_grad()
def evaluate_ndcg(model, data, device, batch_size, k=10):
    """
    Computes NDCG@k for:
      - SASRec
      - BERT4Rec
      - TiSASRec
      - Linear SASRec
    """
    model.eval()
    ndcg_scores = []

    is_tisasrec = model.__class__.__name__ == "TiSASRec"

    for batch in _batch_iter(data, batch_size):
        # -----------------------------
        # Unpack batch
        # -----------------------------
        if is_tisasrec:
            seqs, times, targets = batch
            seqs = seqs.to(device)
            times = times.to(device)
            targets = targets.to(device)
            logits = model(seqs, times)
        else:
            # SASRec / BERT4Rec / LinearSASRec
            if len(batch) == 3:
                seqs, _, targets = batch  # ignore time
            else:
                seqs, targets = batch

            seqs = seqs.to(device)
            targets = targets.to(device)
            logits = model(seqs)

        # -----------------------------
        # Top-k ranking
        # -----------------------------
        topk = torch.topk(logits, k=k, dim=1).indices

        for i in range(targets.size(0)):
            target = targets[i].item()
            preds = topk[i].tolist()

            if target in preds:
                rank = preds.index(target)
                ndcg_scores.append(1.0 / np.log2(rank + 2))
            else:
                ndcg_scores.append(0.0)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def _batch_iter(data, batch_size):
    """
    data elements:
      - SASRec / BERT4Rec / LinearSASRec: (seq_list, target) OR (seq_list, time_list, target)
      - TiSASRec: (seq_list, time_list, target)

    Always converts lists -> tensors.
    """
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        cols = list(zip(*batch))

        tensors = []
        for col in cols:
            tensors.append(torch.tensor(col, dtype=torch.long))

        yield tuple(tensors)

import time
import yaml
import torch
import numpy as np

from data.dataset import SequentialDataset
from models.sasrec import SASRec
from models.bert4rec import BERT4Rec
from models.tisasrec import TiSASRec


def load_model_and_data(config_path, device):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset = SequentialDataset(config)
    data = dataset.get_validation_data()  # Phase 3 uses validation data

    model_name = config["model"]["name"]

    if model_name == "sasrec":
        model = SASRec(
            num_items=dataset.num_items,
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
        )

    elif model_name == "bert4rec":
        model = BERT4Rec(
            num_items=dataset.num_items,
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
            max_seq_len=config["dataset"]["max_seq_len"],
        )

    elif model_name == "tisasrec":
        model = TiSASRec(
            num_items=dataset.num_items,
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
            max_seq_len=config["dataset"]["max_seq_len"],
            num_time_bins=config["model"]["num_time_bins"],
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(device)
    model.eval()

    return model, data, config


def make_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        cols = list(zip(*batch))

        tensors = []
        for col in cols:
            tensors.append(torch.tensor(col, dtype=torch.long))

        yield tuple(tensors)


@torch.no_grad()
def main(config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, data, config = load_model_and_data(config_path, device)
    batch_size = config["evaluation"]["inference_batch_size"]

    batches = list(make_batches(data, batch_size))

    # -----------------------------
    # Warm-up
    # -----------------------------
    for _ in range(5):
        batch = batches[0]
        if len(batch) == 3:
            seqs, times, _ = batch
            model(seqs.to(device), times.to(device))
        else:
            seqs, _ = batch
            model(seqs.to(device))

    # -----------------------------
    # Timed inference
    # -----------------------------
    latencies = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()

    for batch in batches:
        t0 = time.time()

        if len(batch) == 3:
            seqs, times, _ = batch
            model(seqs.to(device), times.to(device))
        else:
            seqs, _ = batch
            model(seqs.to(device))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies.append((time.time() - t0) * 1000)

    total_time = time.time() - start

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_mem = 0.0

    latencies = np.array(latencies)

    results = {
        "avg_latency_ms": latencies.mean(),
        "p95_latency_ms": np.percentile(latencies, 95),
        "throughput_users_per_sec": len(data) / total_time,
        "peak_inference_mem_mb": peak_mem,
    }

    print(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)

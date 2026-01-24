import time
import yaml
import torch
import numpy as np
from data.dataset import SequentialDataset
from models.sasrec import SASRec
from torch.nn.utils.rnn import pad_sequence

def load_model_and_data(config_path, device):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset = SequentialDataset(config)
    test_data = dataset.get_test_data()

    model = SASRec(
        num_items=dataset.num_items,
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
    ).to(device)

    model.eval()
    return model, test_data, config 

def make_batches(data, batch_size):
    sequences = [torch.tensor(seq, dtype=torch.long) for seq, _ in data]

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        padded = pad_sequence(batch, batch_first=True, padding_value=0)
        yield padded

def warmup(model, batches, device, num_warmup=5):
    with torch.no_grad():
        for i, batch in enumerate(batches):
            if i >= num_warmup:
                break
            batch = batch.to(device)
            _ = model(batch)

def measure_inference(model, batches, device):
    latencies = []

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)

            torch.cuda.synchronize()
            start = time.time()

            _ = model(batch)

            torch.cuda.synchronize()
            end = time.time()

            latencies.append(end - start)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return latencies, peak_mem

def summarize(latencies, batch_size):
    latencies = np.array(latencies)

    avg_latency_ms = latencies.mean() * 1000 / batch_size
    p95_latency_ms = np.percentile(latencies, 95) * 1000 / batch_size
    throughput = batch_size / latencies.mean()

    return avg_latency_ms, p95_latency_ms, throughput

def main(config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, test_data, config = load_model_and_data(config_path, device)

    batch_size = config["evaluation"].get("inference_batch_size", 32)

    batches = list(make_batches(test_data, batch_size))

    warmup(model, batches, device)

    latencies, peak_mem = measure_inference(model, batches, device)

    avg_ms, p95_ms, throughput = summarize(latencies, batch_size)

    print(f"Avg latency: {avg_ms:.3f} ms")
    print(f"P95 latency: {p95_ms:.3f} ms")
    print(f"Throughput: {throughput:.2f} users/sec")
    print(f"Peak inference memory: {peak_mem:.1f} MB")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)
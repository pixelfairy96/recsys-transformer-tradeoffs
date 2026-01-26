import argparse
import random
import yaml
import torch
import numpy as np

from data.dataset import SequentialDataset
from training.train_loop import train_one_epoch, evaluate
from training.logger import (
    create_run_dir,
    save_config,
    save_env_info,
    save_metrics,
    get_env_info,
)


def main(config_path):
    # -----------------------------
    # Load config
    # -----------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # -----------------------------
    # Set random seeds
    # -----------------------------
    SEED = 42
    config["seed"] = SEED

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # -----------------------------
    # Create run directory + logs
    # -----------------------------
    run_id, run_dir = create_run_dir()
    save_config(config, run_dir)

    env_info = get_env_info()
    save_env_info(env_info, run_dir)

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = SequentialDataset(config)
    train_data = dataset.get_train_data()
    val_data = dataset.get_validation_data()

    # -----------------------------
    # Model selection
    # -----------------------------
    model_name = config["model"].get("name", "sasrec").lower()

    if model_name == "sasrec":
        from models.sasrec import SASRec

        model = SASRec(
            num_items=dataset.num_items,
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
        ).to(device)

    elif model_name == "bert4rec":
        from models.bert4rec import BERT4Rec

        model = BERT4Rec(
            num_items=dataset.num_items,
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
            max_seq_len=config["dataset"]["max_seq_len"],
        ).to(device)

    elif model_name == "tisasrec":
        from models.tisasrec import TiSASRec

        model = TiSASRec(
            num_items=dataset.num_items,
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
            max_seq_len=config["dataset"]["max_seq_len"],
            num_time_bins=config["model"]["num_time_bins"],
        ).to(device)

    elif model_name == "linear_sasrec":
        from models.linear_sasrec import LinearSASRec
        model = LinearSASRec(
            num_items=dataset.num_items,
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
            max_seq_len=config["dataset"]["max_seq_len"],
        ).to(device)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # -----------------------------
    # Optimizer
    # -----------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
    )

    # -----------------------------
    # Metrics container
    # -----------------------------
    all_metrics = {
        "train": [],
        "validation": [],
    }

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        train_loss, epoch_time, peak_mem = train_one_epoch(
            model,
            train_data,
            optimizer,
            device,
            config["training"]["batch_size"],
        )

        model.eval()
        with torch.no_grad():
            val_acc = evaluate(
                model,
                val_data,
                device,
                config["training"]["batch_size"],
            )

        all_metrics["train"].append({
            "epoch": epoch + 1,
            "loss": train_loss,
            "epoch_time_sec": epoch_time,
            "peak_mem_mb": peak_mem,
        })

        all_metrics["validation"].append({
            "epoch": epoch + 1,
            "val_acc": val_acc,
        })

        print(
            f"Epoch {epoch+1}: "
            f"train loss = {train_loss:.4f}, "
            f"time = {epoch_time:.2f}s, "
            f"peak_mem = {peak_mem:.1f} MB, "
            f"val_ndcg@10 = {val_acc:.4f}"
        )

    # -----------------------------
    # Save metrics once
    # -----------------------------
    save_metrics(all_metrics, run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    main(args.config)

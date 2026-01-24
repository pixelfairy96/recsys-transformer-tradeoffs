import yaml
import torch

from data.dataset import SequentialDataset
from models.sasrec import SASRec
from training.train_loop import train_one_epoch, evaluate


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SequentialDataset(config)

    model = SASRec(
        num_items=dataset.num_items,
        hidden_size=config["model"].get("hidden_size", 64),
        num_layers=config["model"].get("num_layers", 2),
        num_heads=config["model"].get("num_heads", 2),
        dropout=config["model"].get("dropout", 0.2),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"].get("learning_rate", 1e-3)
    )

    train_data = dataset.get_train_data()
    val_data = dataset.get_validation_data()

    for epoch in range(config["training"]["max_epochs"]):
        train_loss, epoch_time, peak_mem = train_one_epoch(
            model,
            train_data,
            optimizer,
            device,
            batch_size=config["training"]["batch_size"]
        )

        val_acc = evaluate(
            model,
            val_data,
            device,
            batch_size=config["training"]["batch_size"]
        )

        print(
            f"Epoch {epoch+1}: "
            f"train loss = {train_loss:.4f}, "
            f"time = {epoch_time:.2f}s, "
            f"peak_mem = {peak_mem:.1f} MB, "
            f"val_acc = {val_acc:.4f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)

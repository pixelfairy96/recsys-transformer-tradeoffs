import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class SequentialDataset:
    def __init__(self, config):
        self.config = config
        self.data_path = config["dataset"]["data_path"]
        self.max_seq_len = config["dataset"]["max_seq_len"]
        self.num_time_bins = config["dataset"].get("num_time_bins", 50)

        self._load_data()
        self._build_sequences()

    def _load_data(self):
        user_interactions = defaultdict(list)
        all_items = set()

        with open(self.data_path, "r") as f:
            for line in f:
                user, item, rating, timestamp = line.strip().split("::")
                user = int(user)
                item = int(item)
                timestamp = int(timestamp)

                user_interactions[user].append((item, timestamp))
                all_items.add(item)

        self.user_interactions = user_interactions
        self.num_items = max(all_items) + 1

    def _discretize_time_diffs(self, diffs):
        diffs = np.array(diffs)
        diffs = np.clip(diffs, 0, diffs.max() if diffs.max() > 0 else 1)
        bins = np.linspace(0, diffs.max() + 1, self.num_time_bins)
        return np.digitize(diffs, bins)

    def _build_sequences(self):
        self.train_data = []
        self.val_data = []

        for user, interactions in self.user_interactions.items():
            interactions.sort(key=lambda x: x[1])

            items = [x[0] for x in interactions]
            times = [x[1] for x in interactions]

            if len(items) < 2:
                continue

            time_diffs = [0]
            for i in range(1, len(times)):
                time_diffs.append(times[i] - times[i - 1])

            time_bins = self._discretize_time_diffs(time_diffs).tolist()

            for i in range(1, len(items)):
                seq_items = items[:i]
                seq_times = time_bins[:i]
                target = items[i]

                seq_items = seq_items[-self.max_seq_len:]
                seq_times = seq_times[-self.max_seq_len:]

                pad_len = self.max_seq_len - len(seq_items)
                seq_items = [0] * pad_len + seq_items
                seq_times = [0] * pad_len + seq_times

                self.train_data.append(
                    (seq_items, seq_times, target)
                )

            # last interaction for validation
            self.val_data.append(
                (seq_items, seq_times, target)
            )

    def _make_loader(self, data, batch_size, shuffle):
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        seqs, times, targets = zip(*batch)
        return (
            torch.tensor(seqs, dtype=torch.long),
            torch.tensor(times, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )

    def get_train_data(self):
        return self.train_data

    def get_validation_data(self):
        return self.val_data

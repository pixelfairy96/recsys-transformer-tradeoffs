import os
from collections import defaultdict
import numpy as np


class SequentialDataset:
    def __init__(self, config):
        """
        Args:
            config (dict): dataset-related configuration
        """
        self.dataset_name = config["dataset"]["name"]
        self.sequence_length = config["dataset"]["sequence_length"]

        if self.dataset_name != "movielens-1m":
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.data_dir = "data/raw/ml-1m"
        self._load_data()
        self._build_sequences()

    def _load_data(self):
        ratings_file = os.path.join(self.data_dir, "ratings.dat")
        if not os.path.exists(ratings_file):
            raise FileNotFoundError(f"Missing file: {ratings_file}")

        user_interactions = defaultdict(list)

        with open(ratings_file, "r") as f:
            for line in f:
                user_id, item_id, _, timestamp = line.strip().split("::")
                user_interactions[int(user_id)].append(
                    (int(item_id), int(timestamp))
                )

        # sort interactions by time
        for user in user_interactions:
            user_interactions[user].sort(key=lambda x: x[1])

        self.user_sequences = {
            user: [item for item, _ in interactions]
            for user, interactions in user_interactions.items()
            if len(interactions) >= 3
        }

    def _build_sequences(self):
        self.train_data = []
        self.val_data = []
        self.test_data = []

        all_items = set()

        for user, seq in self.user_sequences.items():
            all_items.update(seq)

            train_seq = seq[:-2]
            val_item = seq[-2]
            test_item = seq[-1]

            train_seq = train_seq[-self.sequence_length :]

            self.train_data.append((train_seq, val_item))
            self.val_data.append((train_seq, val_item))
            self.test_data.append((train_seq, test_item))

        self.num_items_value = max(all_items) + 1  # padding index = 0

    def get_train_data(self):
        return self.train_data

    def get_validation_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data

    @property
    def num_items(self):
        return self.num_items_value
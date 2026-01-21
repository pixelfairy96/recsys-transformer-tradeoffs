"""
Dataset utilities for sequential recommendation.

This module defines a unified dataset interface used by all models.
"""

class SequentialDataset:
    def __init__(self, config):
        """
        Args:
            config (dict): dataset-related configuration
        """
        raise NotImplementedError

    def get_train_data(self):
        """
        Returns:
            train_sequences: list or tensor of training sequences
        """
        raise NotImplementedError

    def get_validation_data(self):
        """
        Returns:
            validation_sequences
        """
        raise NotImplementedError

    def get_test_data(self):
        """
        Returns:
            test_sequences
        """
        raise NotImplementedError

    @property
    def num_items(self):
        """
        Returns:
            int: number of unique items
        """
        raise NotImplementedError

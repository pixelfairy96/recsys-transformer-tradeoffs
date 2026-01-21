"""
Evaluation metrics for sequential recommendation.

All models are evaluated using the same metrics and protocol.
"""

def evaluate_ranking(scores, targets, k_values):
    """
    Compute ranking-based evaluation metrics.

    Args:
        scores: predicted scores for items (shape: [batch_size, num_items])
        targets: ground-truth item indices (shape: [batch_size])
        k_values: list of K values for evaluation (e.g., [10, 20])

    Returns:
        dict: mapping metric names to values
    """
    raise NotImplementedError
import torch
import pytest

from src.utils import evaluate_classification


@pytest.mark.order(13)
def test_evaluate_classification():
    # Given
    predictions = torch.tensor([1, 0, 1, 1, 0])
    labels = torch.tensor([1, 0, 0, 1, 1])

    # Expected metrics
    expected_accuracy = 3 / 5  # 3 correct predictions out of 5
    expected_precision = 2 / 3  # 2 true positives, 1 false positive
    expected_recall = 2 / 3  # 2 true positives, 1 false negative
    expected_f1_score = (
        2
        * (expected_precision * expected_recall)
        / (expected_precision + expected_recall)
    )

    # When
    metrics = evaluate_classification(predictions, labels)

    if metrics is None:
        pytest.skip()
    
    # Then
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert metrics["accuracy"] == expected_accuracy
    assert metrics["precision"] == expected_precision
    assert metrics["recall"] == expected_recall
    assert metrics["f1_score"] == expected_f1_score

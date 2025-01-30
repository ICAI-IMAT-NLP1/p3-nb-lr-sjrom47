import pytest
import torch
from src.naive_bayes import NaiveBayes


@pytest.mark.order(4)
def test_estimate_class_priors():
    # Given
    labels = torch.tensor([0, 1, 0], dtype=torch.int)
    model = NaiveBayes()
    priors = model.estimate_class_priors(labels)

    # Then
    assert isinstance(priors, dict)
    assert set(priors.keys()) == {0, 1}
    assert priors[0] == 2 / 3
    assert priors[1] == 1 / 3


@pytest.mark.order(5)
def test_estimate_conditional_probabilities():
    # Given
    features = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.int)
    model = NaiveBayes()
    model.fit(features, labels, delta=1.0)

    # When
    cond_probs = model.conditional_probabilities

    # Then
    assert isinstance(cond_probs, dict)
    assert all(isinstance(cond_probs[label], torch.Tensor) for label in cond_probs)
    assert torch.allclose(cond_probs[0], torch.tensor([2 / 5, 2 / 5, 1 / 5], dtype=torch.float32))
    assert torch.allclose(cond_probs[1], torch.tensor([1 / 5, 2 / 5, 2 / 5], dtype=torch.float32))
    



@pytest.mark.order(6)
def test_predict_naive_bayes():
    # Given
    features = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.int)
    model = NaiveBayes()
    model.fit(features, labels, delta=1.0)
    test_feature = torch.tensor([1, 0, 1], dtype=torch.float32)

    # When
    predicted_class = model.predict(test_feature)

    # Then
    assert isinstance(predicted_class, int)
    assert predicted_class == 0

    test_feature = torch.tensor([0, 0, 1], dtype=torch.float32)
    # When
    predicted_class = model.predict(test_feature)

    # Then
    assert isinstance(predicted_class, int)
    assert predicted_class == 1


@pytest.mark.order(7)
def test_predict_proba_naive_bayes():
    # Given
    features = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.int)
    model = NaiveBayes()
    model.fit(features, labels, delta=1.0)
    test_feature = torch.tensor([1, 0, 1], dtype=torch.float32)

    # When
    probabilities = model.predict_proba(test_feature)

    # Then
    assert isinstance(probabilities, torch.Tensor)
    assert torch.allclose(probabilities, torch.tensor([0.5, 0.5], dtype=torch.float32), atol=1e-2)
    assert probabilities.dim() == 1
    assert torch.sum(probabilities).item() == pytest.approx(1.0)

    test_feature = torch.tensor([0, 0, 1], dtype=torch.float32)
    # When
    probabilities = model.predict_proba(test_feature)

    # Then
    assert torch.allclose(probabilities, torch.tensor([0.3333, 0.6667], dtype=torch.float32), atol=1e-2)
    assert torch.sum(probabilities).item() == pytest.approx(1.0)

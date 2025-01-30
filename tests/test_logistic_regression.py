import pytest
import torch
from src.logistic_regression import LogisticRegression


@pytest.mark.order(8)
def test_sigmoid():
    # Given
    input_tensor = torch.tensor([0, 2, -2], dtype=torch.float32)

    random_state = 42  # Example seed for reproducibility
    model = LogisticRegression(random_state=random_state)

    # When
    sigmoid_values = model.sigmoid(input_tensor)

    # Then
    assert sigmoid_values.shape == input_tensor.shape
    assert torch.all(sigmoid_values >= 0) and torch.all(sigmoid_values <= 1)
    # Checking sigmoid of 0 is 0.5
    assert torch.isclose(sigmoid_values[0], torch.tensor(0.5))


@pytest.mark.order(9)
def test_binary_cross_entropy_loss():
    # Given
    predictions = torch.tensor([0.9, 0.3, 0.2], dtype=torch.float32)
    targets = torch.tensor([1, 0, 0], dtype=torch.float32)

    random_state = 42  # Example seed for reproducibility
    model = LogisticRegression(random_state=random_state)

    # When
    loss = model.binary_cross_entropy_loss(predictions, targets)

    # Then
    assert loss >= 0  # Loss should always be non-negative
    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, torch.tensor(0.2284, dtype=torch.float32), atol=1e-2)


@pytest.mark.order(10)
def test_train_logistic_regression():
    # Given a small, controlled dataset
    features = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 1], dtype=torch.float32)
    learning_rate = 0.01
    epochs = 100
    random_state = 42
    model = LogisticRegression(random_state=random_state)

    # When training the model
    model.fit(features, labels, learning_rate, epochs)

    # Then, check if weights are updated
    assert model.weights is not None
    assert torch.allclose(model.weights, torch.tensor([0.2222, 0.0288, 0.4485, 0.2301], dtype=torch.float32), atol=1e-2)


@pytest.mark.order(11)
def test_predict_logistic_regression():
    # Given
    test_features = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
    random_state = 42
    model = LogisticRegression(random_state=random_state)
    model.weights = torch.tensor(
        [0.1, -0.2, 0.3, 0.4], dtype=torch.float32
    )  # Including bias weight

    # When
    predictions = model.predict(test_features)

    # Then
    assert predictions.shape[0] == test_features.shape[0]
    assert torch.all((predictions == 0) | (predictions == 1))


@pytest.mark.order(12)
def test_predict_proba_logistic_regression():
    # Given
    test_features = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
    random_state = 42
    model = LogisticRegression(random_state=random_state)
    model.weights = torch.tensor(
        [0.1, -0.2, 0.3, 0.4], dtype=torch.float32
    )  # Including bias weight

    # When
    probabilities = model.predict_proba(test_features)

    # Then
    assert probabilities.shape[0] == test_features.shape[0]
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert torch.allclose(probabilities, torch.tensor([0.69, 0.6225], dtype=torch.float32), atol=1e-2)

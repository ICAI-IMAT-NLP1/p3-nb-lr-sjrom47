import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = len(
            features[0]
        )  # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(
            features, labels, delta
        )
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        label_counts: Counter = Counter(list(labels.numpy()))
        class_priors: Dict[int, torch.Tensor] = {
            int(label): torch.tensor(label_counts[label] / len(labels))
            for label in label_counts
        }
        if class_priors is None:
            raise ValueError("No class priors were estimated.")
        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        unique_labels = torch.unique(labels)
        total_counts: Dict[int, torch.Tensor] = {}
        class_word_counts: Dict[int, torch.Tensor] = {}

        for lbl in unique_labels:
            # Boolean mask for current label
            mask = labels == lbl
            # Sum word counts for all samples of the current label
            word_sum = features[mask, :].sum(dim=0)
            total_counts[int(lbl)] = word_sum
            # Apply smoothing to get probabilities for each word in the class
            class_word_counts[int(lbl)] = (word_sum + delta) / (
                word_sum.sum() + delta * self.vocab_size
            )

        return class_word_counts

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        log_posteriors: torch.Tensor = torch.tensor(
            [
                (feature * torch.log(cond_probs)).sum(dim=0) + torch.log(class_prior)
                for cond_probs, class_prior in zip(
                    self.conditional_probabilities.values(), self.class_priors.values()
                )
            ]
        )
        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        log_posteriors: torch.Tensor = self.estimate_class_posteriors(feature)
        pred: int = int(torch.argmax(log_posteriors).item())
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        log_posteriors: torch.Tensor = self.estimate_class_posteriors(feature)
        probs: torch.Tensor = torch.nn.functional.softmax(log_posteriors, dim=0)
        return probs

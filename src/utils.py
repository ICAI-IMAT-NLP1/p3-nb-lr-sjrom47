from typing import List, Dict
import torch
import re
import string


def accuracy_score(y_true, y_pred):
    """To calculate the accuracy score of the model"""
    return (y_true == y_pred).sum() / len(y_true)


def recall_score(y_true, y_pred):
    """To calculate the recall score of the model"""
    true_positives = (y_true * y_pred).sum()
    actual_positives = y_true.sum()
    return true_positives / actual_positives


def precision_score(y_true, y_pred):
    """To calculate the precision score of the model"""
    true_positives = (y_true * y_pred).sum()
    predicted_positives = y_pred.sum()
    return true_positives / predicted_positives


def f1_score(y_true, y_pred):
    """To calculate the f1 score of the model"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


def remove_punctuations(input_col):
    """To remove all the punctuations present in the text.Input the text column"""
    table = str.maketrans("", "", string.punctuation)
    return input_col.translate(table)


# Tokenizes a input_string. Takes a input_string (a sentence), splits out punctuation and contractions, and returns a list of
# strings, with each input_string being a token.
def tokenize(input_string):
    input_string = remove_punctuations(input_string)
    input_string = re.sub(r"[^A-Za-z0-9(),.!?\'`\-\"]", " ", input_string)
    input_string = re.sub(r"\'s", " 's", input_string)
    input_string = re.sub(r"\'ve", " 've", input_string)
    input_string = re.sub(r"n\'t", " n't", input_string)
    input_string = re.sub(r"\'re", " 're", input_string)
    input_string = re.sub(r"\'d", " 'd", input_string)
    input_string = re.sub(r"\'ll", " 'll", input_string)
    input_string = re.sub(r"\.", " . ", input_string)
    input_string = re.sub(r",", " , ", input_string)
    input_string = re.sub(r"!", " ! ", input_string)
    input_string = re.sub(r"\?", " ? ", input_string)
    input_string = re.sub(r"\(", " ( ", input_string)
    input_string = re.sub(r"\)", " ) ", input_string)
    input_string = re.sub(r"\-", " - ", input_string)
    input_string = re.sub(r"\"", ' " ', input_string)
    # We may have introduced double spaces, so collapse these down
    input_string = re.sub(r"\s{2,}", " ", input_string)
    return list(filter(lambda x: len(x) > 0, input_string.split(" ")))


class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (List[str]): List of words.
        label (int): Sentiment label (0 for negative, 1 for positive).
    """

    def __init__(self, words: List[str], label: int):
        self._words = words
        self._label = label

    def __repr__(self) -> str:
        if self.label is not None:
            return f"{self.words}; label={self.label}"
        else:
            return f"{self.words}, no label"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, SentimentExample):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.words == other.words and self.label == other.label

    @property
    def words(self):
        return self._words

    @words.setter
    def words(self, value):
        raise NotImplemented

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        raise NotImplemented


def evaluate_classification(
    predictions: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate classification metrics including accuracy, precision, recall, and F1-score.

    Args:
        predictions (torch.Tensor): Predictions from the model.
        labels (torch.Tensor): Actual ground truth labels.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    metrics_functions: List[function] = [
        accuracy_score,
        recall_score,
        precision_score,
        f1_score,
    ]
    metrics_names: List[str] = [
        f.__name__.split("_")[0] if not f.__name__.startswith("f1") else f.__name__
        for f in metrics_functions
    ]
    metrics: Dict[str, float] = {
        metric_name: metric(labels, predictions)
        for metric_name, metric in zip(metrics_names, metrics_functions)
    }

    return metrics

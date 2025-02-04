from typing import List, Dict, Tuple
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    with open(infile, "r", encoding="utf-8") as f:
        lines: List[str] = f.readlines()
        sentences: Tuple[str]
        labels: Tuple[str]
        sentences, labels = zip(*[line.rsplit(sep="\t", maxsplit=1) for line in lines])
        examples: List[SentimentExample] = [
            SentimentExample(tokenize(sentence), int(label))
            for sentence, label in zip(sentences, labels)
        ]
    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    unique_words: List[str] = list(
        set([word for example in examples for word in example.words])
    )
    vocab: Dict[str, int] = {word: i for i, word in enumerate(unique_words)}

    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    word_counts: Counter = Counter(text)
    bow: torch.Tensor = torch.zeros(len(vocab))
    for word in word_counts:
        if word in vocab:
            if binary:
                bow[vocab[word]] = 1
            else:
                bow[vocab[word]] = word_counts[word]

    return bow

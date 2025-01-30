from typing import List
import torch
import pytest

from src.data_processing import (
    read_sentiment_examples,
    build_vocab,
    bag_of_words,
)
from src.utils import SentimentExample


@pytest.mark.order(1)
def test_read_sentiment_examples():
    # Given a sample file with known content
    test_file = "data/sample_train.txt"

    # When the function is called
    examples = read_sentiment_examples(test_file)

    if examples is None:
        pytest.skip()

    # Then it should return a list of SentimentExample objects
    assert isinstance(examples, list)
    assert all(isinstance(ex, SentimentExample) for ex in examples)

    # And the contents should match the expected results
    expected_results = [
        SentimentExample(["Example", "sentence", "one"], 0),
        SentimentExample(["Another", "example", "sentence"], 1),
    ]
    assert all([ex == exp_ex for ex, exp_ex in zip(examples, expected_results)])


@pytest.mark.order(2)
def test_build_vocab():
    # Given a list of SentimentExample objects with known content
    examples = [
        SentimentExample(["word1", "word2"], 0),
        SentimentExample(["word2", "word3"], 1),
    ]

    # When the function is called
    vocab = build_vocab(examples)

    if vocab is None:
        pytest.skip()

    # Then it should return a dictionary
    assert isinstance(vocab, dict)

    # And the dictionary should have unique indices for each word
    assert len(vocab) == 3  # Ensure there are 3 unique words
    assert all(isinstance(index, int) for word, index in vocab.items())

    # And the words in the examples should be in the vocabulary
    expected_words = ["word1", "word2", "word3"]
    assert all(word in vocab for word in expected_words)

    # Check if the indices are sequential and start from 0
    assert set(vocab.values()) == set(range(3))


@pytest.mark.order(3)
def test_bag_of_words_full():
    # Given a known vocabulary
    vocab = {"word1": 0, "word2": 1, "word3": 2}

    # And a sample text
    text = ["word1", "word2", "word1", "word4"]  # Note: 'word4' is not in vocab

    # When the function is called for full BoW
    vector = bag_of_words(text, vocab, binary=False)

    if vector is None:
        pytest.skip()
        
    # Then it should correctly represent the word counts in full BoW
    expected_vector = torch.tensor(
        [2, 1, 0], dtype=torch.float32
    )  # 'word1': 2, 'word2': 1, 'word3': 0
    assert torch.equal(vector, expected_vector)


@pytest.mark.order(3)
def test_bag_of_words_binary():
    # Given the same known vocabulary and sample text
    vocab = {"word1": 0, "word2": 1, "word3": 2}
    text = ["word1", "word2", "word1", "word4"]

    # When the function is called for binary BoW
    vector = bag_of_words(text, vocab, binary=True)

    if vector is None:
        pytest.skip()

    # Then it should correctly represent the word presence in binary BoW
    expected_vector = torch.tensor([1, 1, 0], dtype=torch.float32)
    assert torch.equal(vector, expected_vector)

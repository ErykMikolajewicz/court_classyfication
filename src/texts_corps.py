import random
import pickle
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
from typing import Literal

import numpy as np

from src.labeling import get_object_label


def _get_counter_object(counter_path: Path) -> Counter:
    with open(counter_path, 'rb') as counter_file:
        counter = pickle.load(counter_file)
    return counter


def get_vocabulary(fraction: float = 0.8, document_frequency_threshold: int = 5) -> dict[str, SimpleNamespace]:
    counters_number = get_counters_number('training')
    number_to_make_vocabulary = int(counters_number * fraction)

    random.seed(42)

    counter_dir = Path('data') / 'counters' / 'training'
    counters_paths = list(counter_dir.iterdir())
    selected_counters_paths = random.sample(counters_paths, k=number_to_make_vocabulary)

    document_words_frequency_count = Counter()
    for counter_path in selected_counters_paths:
        counter = _get_counter_object(counter_path)
        document_words_frequency_count.update(counter)

    vocabulary = {}
    word_index = 1  # 0 reserved for unknown words
    for word, count in document_words_frequency_count.items():
        if count > document_frequency_threshold:
            vocabulary[word] = SimpleNamespace(index=word_index, count=count)
            word_index += 1

    return vocabulary


def get_cases_words_count(set_type: str, label_type: Literal["detailed", "general"]) -> (Counter, str):
    counter_dir = Path('data') / 'counters' / set_type
    for counter_path in counter_dir.iterdir():
        counter = _get_counter_object(counter_path)
        counter_file_name = counter_path.name
        label = get_object_label(counter_file_name, label_type)
        yield counter, label


def get_counters_number(set_type: str) -> int:
    counter_dir = Path('data') / 'counters' / set_type
    counters_number =  len(list(counter_dir.iterdir()))
    return counters_number


def get_tokens_with_label(set_type: str, label_type: Literal["detailed", "general"]) -> (np.array, str):
    tokens_dir = Path('data') / 'tokens' / set_type
    for tokens_path in tokens_dir.iterdir():
        with open(tokens_path, 'rb') as tokens_file:
            tokens = pickle.load(tokens_file)
        tokens_file_name = tokens_path.name
        label = get_object_label(tokens_file_name, label_type)
        yield tokens, label


def get_tokens_arrays_number(set_type: str) -> int:
    tokens_dir = Path('data') / 'tokens' / set_type
    tokens_arrays_number =  len(list(tokens_dir.iterdir()))
    return tokens_arrays_number

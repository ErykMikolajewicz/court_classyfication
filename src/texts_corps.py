import random
import pickle
from pathlib import Path
from collections import Counter
from typing import Literal
from collections import defaultdict

from src.labeling import get_counter_label


def _get_counter_object(counter_path: Path) -> Counter:
    with open(counter_path, 'rb') as counter_file:
        counter = pickle.load(counter_file)
    return counter


def get_vocabulary(set_name: Literal['training', 'validation'], fraction: float = 0.8,
                   document_frequency_threshold: int = 5) -> dict:
    counters_dir = Path('data') / set_name / 'counters'
    counters_paths = [path for path in counters_dir.iterdir()]
    counters_number = len(counters_paths)
    number_to_make_vocabulary = int(counters_number * fraction)

    random.seed(42)
    selected_counters_paths = random.sample(counters_paths, k=number_to_make_vocabulary)

    document_words_frequency_count = defaultdict(lambda: 0)
    for counter_path in selected_counters_paths:
        counter = _get_counter_object(counter_path)
        for word in counter.keys():
            document_words_frequency_count[word] += 1

    vocabulary = {}
    n = 1  # 0 reserved for unknown words
    for word, count in document_words_frequency_count.items():
        if count > document_frequency_threshold:
            vocabulary[word] = n
            n += 1

    return vocabulary


def get_cases_words_count(set_name: Literal['training', 'validation']) -> Counter:
    counters_dir = Path('data') / set_name / 'counters'
    counters_paths = [path for path in counters_dir.iterdir()]
    for counter_path in counters_paths:
        counter = _get_counter_object(counter_path)
        counter_file_name = counter_path.name
        label = get_counter_label(counter_file_name)
        yield counter, label


def get_counters_number(set_name: Literal['training', 'validation']) -> int:
    counters_dir = Path('data') / set_name / 'counters'
    counters_paths = [path for path in counters_dir.iterdir()]
    counters_number = len(counters_paths)
    return counters_number

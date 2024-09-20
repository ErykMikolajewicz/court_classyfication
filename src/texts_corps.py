import random
import pickle
from pathlib import Path
from collections import Counter

from src.labeling import get_counter_label

counters_dir = Path('./data/counters')
counters_paths = [path for path in counters_dir.iterdir()]
counters_number = len(counters_paths)


def _get_counter_object(counter_path: Path) -> Counter:
    with open(counter_path, 'rb') as counter_file:
        counter = pickle.load(counter_file)
    return counter


def get_vocabulary(fraction: float = 0.8) -> dict:
    number_to_make_vocabulary = int(counters_number * fraction)

    random.seed(42)  # Actually not working as excepted, random.sample have bug
    selected_counters_paths = random.sample(counters_paths, k=number_to_make_vocabulary)

    vocabulary = set()
    for counter_path in selected_counters_paths:
        counter = _get_counter_object(counter_path)
        case_words = counter.keys()
        vocabulary.update(case_words)

    vocabulary_with_numbers = {}
    for n, word in enumerate(vocabulary, start=1):  # 0 reserved for unknown words
        vocabulary_with_numbers[word] = n

    return vocabulary_with_numbers


def get_cases_words_count() -> Counter:
    for counter_path in counters_paths:
        counter = _get_counter_object(counter_path)
        counter_file_name = counter_path.name
        label = get_counter_label(counter_file_name)
        yield counter, label


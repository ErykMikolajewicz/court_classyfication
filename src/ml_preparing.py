from typing import Literal

import numpy as np

from src.texts_corps import get_cases_words_count, get_counters_number, get_tokens_with_label, get_tokens_arrays_number


def get_bag_unknown(vocabulary: dict, set_type: str, label_type: Literal["detailed", "general"],
                    max_size=None) -> (np.ndarray, np.ndarray):
    vocabulary_length = len(vocabulary)
    counters_number = get_counters_number(set_type)

    if max_size:
        data_length = max_size
    else:
        data_length = counters_number
    features = np.zeros((data_length, vocabulary_length + 1), dtype=np.int32)
    target = np.empty((data_length, 1), dtype=object)

    for row_index, (words_count, label) in enumerate(get_cases_words_count(set_type, label_type)):
        if max_size:
            if row_index >= max_size:
                break

        target[row_index] = label
        for word, number_of_word_in_document in words_count.items():
            try:
                column_index = vocabulary[word].index
            except KeyError:
                features[row_index, 0] += number_of_word_in_document
                continue
            features[row_index, column_index] = number_of_word_in_document

    return features, target


def get_tokens(set_type: str, label_type: Literal["detailed", "general"], max_tokens: int,
               max_size=None) -> (np.ndarray, np.ndarray):

    tokens_arrays_number = get_tokens_arrays_number(set_type)

    if max_size:
        data_length = max_size
    else:
        data_length = tokens_arrays_number
    features = np.zeros((data_length, max_tokens), dtype=np.int32)
    target = np.empty((data_length, 1), dtype=object)

    for row_index, (token_array, label) in enumerate(get_tokens_with_label(set_type, label_type)):
        if max_size:
            if row_index >= max_size:
                break

        target[row_index] = label
        array_length = len(token_array)
        length_to_copy = min(array_length, max_tokens)
        features[row_index, :length_to_copy] = token_array[:length_to_copy]

    return features, target

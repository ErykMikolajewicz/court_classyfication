import numpy as np

from src.texts_corps import get_vocabulary, get_cases_words_count, counters_number


def main():
    vocabulary = get_vocabulary()
    vocabulary_length = len(vocabulary)

    features_array = np.zeros((counters_number, vocabulary_length + 1), dtype=np.int32)
    for row_index, words_count in enumerate(get_cases_words_count()):
        for word, n in words_count.items():
            try:
                column_index = vocabulary[word]
                features_array[row_index, column_index] = n
            except KeyError:
                features_array[row_index, 0] += n

main()

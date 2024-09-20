import numpy as np
from sklearn.preprocessing import OneHotEncoder

from src.texts_corps import get_vocabulary, get_cases_words_count, counters_number


def main():
    vocabulary = get_vocabulary()
    vocabulary_length = len(vocabulary)

    features = np.zeros((counters_number, vocabulary_length + 1), dtype=np.int32)
    target = np.empty((counters_number, 1), dtype=object)
    for row_index, (words_count, label) in enumerate(get_cases_words_count()):
        target[row_index] = label
        for word, n in words_count.items():
            try:
                column_index = vocabulary[word]
                features[row_index, column_index] = n
            except KeyError:
                features[row_index, 0] += n
    encoder = OneHotEncoder()
    encoder.fit(target)
    target = encoder.transform(target)


main()

from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.texts_corps import get_vocabulary, get_cases_words_count, get_counters_number


def get_features(set_name: Literal['training', 'validation'], vocabulary: dict):
    vocabulary_length = len(vocabulary)
    counters_number = get_counters_number(set_name)
    features = np.zeros((counters_number, vocabulary_length + 1), dtype=np.int32)
    for row_index, (words_count, label) in enumerate(get_cases_words_count(set_name)):
        for word, n in words_count.items():
            try:
                column_index = vocabulary[word]
                features[row_index, column_index] = n
            except KeyError:
                features[row_index, 0] += n
    return features


def get_target(set_name: Literal['training', 'validation']) -> np.array:
    counters_number = get_counters_number(set_name)
    target = np.empty((counters_number, 1), dtype=object)
    for row_index, (_, label) in enumerate(get_cases_words_count(set_name)):
        target[row_index] = label
    return target


def main():
    vocabulary = get_vocabulary('training')

    training_features: np.array = get_features('training', vocabulary)
    training_target = get_target('training')

    encoder = OneHotEncoder()
    encoder.fit(training_target)
    training_target = encoder.transform(training_target)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(5, 500), random_state=42)

    clf.fit(training_features, training_target)

    validation_features = get_features('validation', vocabulary)
    validation_target = get_target('validation')
    validation_target = encoder.transform(validation_target)

    validation_predict = clf.predict(validation_features)
    print(accuracy_score(validation_target, validation_predict))

    cm = confusion_matrix(validation_target.toarray().argmax(axis=1), validation_predict.toarray().argmax(axis=1))

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cywilne", "Praca","Ubezpieczenia Społeczne",
                                                                       "Inne"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()


main()

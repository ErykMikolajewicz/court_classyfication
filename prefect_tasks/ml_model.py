from typing import Literal
import pickle

from prefect import task
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from src.texts_corps import get_vocabulary, get_cases_words_count, get_counters_number


def get_features_and_target(set_name: Literal['training', 'validation'], vocabulary: dict):
    vocabulary_length = len(vocabulary)
    counters_number = get_counters_number(set_name)

    features = np.zeros((counters_number, vocabulary_length + 1), dtype=np.int32)
    target = np.empty((counters_number, 1), dtype=object)
    for row_index, (words_count, label) in enumerate(get_cases_words_count(set_name)):
        target[row_index] = label
        for word, n in words_count.items():
            try:
                column_index = vocabulary[word]
            except KeyError:
                features[row_index, 0] += n
                continue
            features[row_index, column_index] = n
    return features, target


@task
def train_model():
    vocabulary = get_vocabulary('training')

    training_features, training_target = get_features_and_target('training', vocabulary)

    encoder = OneHotEncoder()
    encoder.fit(training_target)
    training_target = encoder.transform(training_target)

    clf = MLPClassifier(random_state=42)
    return

    grid_parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10],
                       'hidden_layer_sizes': [(10, 10), (10, 20), (10, 30), (20, 20), (20, 10), (30, 10)]}
    grid_search = GridSearchCV(clf, grid_parameters, cv=5, n_jobs=8)  # with 8 jobs use 30 Gb Ram on Linux
    grid_search.fit(training_features, training_target)  # Very long computing ~ 7 hours with 8 cores

    print(grid_search.best_params_)
    print(grid_search.best_score_)

    clf = MLPClassifier(random_state=42, **{'alpha': 0.1, 'hidden_layer_sizes': (20, 20)} ) #**grid_search.best_params_

    trained_model = clf.fit(training_features, training_target)
    with open('py_objects/sklearn_best_model.pickle', 'wb') as model_file:
        pickle.dump(trained_model, model_file)


@task
def validate_model():
    vocabulary = get_vocabulary('training')

    with open('py_objects/sklearn_best_model.pickle', 'rb') as model_file:
        clf = pickle.load(model_file)

    validation_features, validation_target = get_features_and_target('validation', vocabulary)

    encoder = OneHotEncoder()
    encoder.fit(validation_target)
    validation_target = encoder.transform(validation_target)

    validation_predict = clf.predict(validation_features)
    print(accuracy_score(validation_target, validation_predict))

    cm = confusion_matrix(validation_target.toarray().argmax(axis=1),
                          validation_predict.toarray().argmax(axis=1))

    labels = np.unique(validation_target)
    sn.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Reds", annot=True, cbar=False, fmt='n')
    plt.xticks(rotation=30)
    plt.title('Confusion Matrix')
    plt.savefig('charts/sklearn_validation_confusion_matrix.png')

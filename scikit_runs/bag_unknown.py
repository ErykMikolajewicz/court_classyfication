"""Implementation of neural network using bag of words, with 0 as index for unknown words.
Generally best model with accuracy 85% on validation dataset."""
import pickle
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sn
from scipy import sparse
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from src.texts_corps import get_vocabulary
from src.ml_preparing import get_bag_unknown
from src.labeling import get_labels


def bag_with_unknown(label_type: Literal["detailed", "general"]):
    vocabulary = get_vocabulary()

    features, target = get_bag_unknown(vocabulary, label_type, max_size=1000)
    features = sparse.csr_matrix(features, dtype=np.int32)

    labels = get_labels(label_type)
    encoder = OneHotEncoder(categories=[labels], dtype=np.int32)
    target = encoder.fit_transform(target)

    training_features, validation_features, training_target, validation_target = train_test_split(
    features, target, test_size=0.2, random_state=42)

    clf = MLPClassifier(random_state=42)

    grid_parameters = {'alpha': [0.01, 0.1, 1],
                       'hidden_layer_sizes': [(10, 10), (10, 20), (10, 30), (20, 20), (20, 10), (30, 10)]}

    grid_search = GridSearchCV(clf, grid_parameters, cv=3, verbose=2)
    grid_search.fit(training_features, training_target)

    print(grid_search.best_params_)
    print(grid_search.best_score_)

    clf = MLPClassifier(random_state=42, **grid_search.best_params_)

    trained_model = clf.fit(training_features, training_target)
    with open('py_objects/sklearn_model.pickle', 'wb') as model_file:
        pickle.dump(trained_model, model_file)

    with open('py_objects/sklearn_model.pickle', 'rb') as model_file:
        clf = pickle.load(model_file)

    validation_predict = clf.predict(validation_features)
    print(accuracy_score(validation_target, validation_predict))

    cm = confusion_matrix(validation_target.toarray().argmax(axis=1), validation_predict.toarray().argmax(axis=1))
    plt.rcParams["pdf.use14corefonts"] = True
    plt.rcParams["ps.useafm"] = True

    sn.heatmap(cm, cmap="Reds", annot=True, cbar=False, fmt='n')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('charts/sklearn_validation_confusion_matrix.png', dpi=300)

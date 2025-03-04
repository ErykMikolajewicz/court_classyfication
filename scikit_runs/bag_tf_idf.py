"""Implementation of neural network, with tf-idf. Do not make accuracy better, achieve 83%"""
import pickle
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from src.texts_corps import get_vocabulary
from src.ml_preparing import get_bag_unknown
from src.labeling import get_labels

def train_tf_idf(label_type: Literal["detailed", "general"]):
    vocabulary = get_vocabulary()
    training_features, training_target = get_bag_unknown(vocabulary, 'training', label_type, max_size=12000)

    tf_idf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    training_features = tf_idf.fit_transform(training_features).toarray()

    labels = get_labels(label_type)
    encoder = OneHotEncoder(categories=[labels])
    training_target = encoder.fit_transform(training_target)

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


def validate_tf_idf(label_type: Literal["detailed", "general"]):
    vocabulary = get_vocabulary()
    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)

    with open('py_objects/sklearn_model.pickle', 'rb') as model_file:
        clf = pickle.load(model_file)

    validation_features, validation_target = get_bag_unknown(vocabulary, 'validation', label_type)

    validation_features = tfidf.fit_transform(validation_features).toarray()

    labels = get_labels(label_type)
    encoder = OneHotEncoder(categories=[labels])
    validation_target = encoder.fit_transform(validation_target)

    validation_predict = clf.predict(validation_features)
    print(accuracy_score(validation_target, validation_predict))

    cm = confusion_matrix(validation_target.toarray().argmax(axis=1), validation_predict.toarray().argmax(axis=1))
    plt.rcParams["pdf.use14corefonts"] = True
    plt.rcParams["ps.useafm"] = True

    sn.heatmap(cm, cmap="Reds", annot=True, cbar=False, fmt='n')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('charts/sklearn_validation_confusion_matrix.png', dpi=300)

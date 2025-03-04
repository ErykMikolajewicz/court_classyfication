"""Bag of words using random forest classifier."""
import pickle
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from src.texts_corps import get_vocabulary
from src.ml_preparing import get_bag_unknown


def train_random_tree(label_type: Literal["detailed", "general"]):
    vocabulary = get_vocabulary()

    training_features, training_target = get_bag_unknown(vocabulary, 'training', label_type, max_size=40_000)

    encoder = OneHotEncoder(sparse_output=False)
    training_target = encoder.fit_transform(training_target)

    rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=30)

    grid_parameters = {'criterion': ['gini', 'entropy'], 'n_estimators': [10, 30, 70, 100, 120, 150]}

    grid_search = GridSearchCV(rf, grid_parameters, cv=5, verbose=2)
    grid_search.fit(training_features, training_target)

    print(grid_search.best_params_)
    print(grid_search.best_score_)

    rf = RandomForestClassifier(random_state=42, **grid_search.best_params_, max_depth=30)

    trained_model = rf.fit(training_features, training_target)
    print([estimator.tree_.max_depth for estimator in rf.estimators_])
    with open('py_objects/sklearn_model.pickle', 'wb') as model_file:
        pickle.dump(trained_model, model_file)


def validate_random_tree(label_type: Literal["detailed", "general"]):
    vocabulary = get_vocabulary()

    validation_features, validation_target = get_bag_unknown(vocabulary, 'validation', label_type)

    encoder = OneHotEncoder(sparse_output=False)
    validation_target = encoder.fit_transform(validation_target)

    with open('py_objects/sklearn_model.pickle', 'rb') as model_file:
        rf = pickle.load(model_file)
    validation_predict = rf.predict(validation_features)
    print(accuracy_score(validation_target, validation_predict))

    cm = confusion_matrix(validation_target.argmax(axis=1), validation_predict.argmax(axis=1))
    plt.rcParams["pdf.use14corefonts"] = True
    plt.rcParams["ps.useafm"] = True

    sn.heatmap(cm, cmap="Reds", annot=True, cbar=False, fmt='n')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('charts/sklearn_validation_confusion_matrix.png', dpi=300)

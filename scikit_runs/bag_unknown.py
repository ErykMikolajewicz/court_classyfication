"""Implementation of neural network using bag of words, with 0 as index for unknown words.
Generally best model with accuracy 85% on validation dataset."""
import pickle

import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from src.texts_corps import get_vocabulary
from src.ml_preparing import get_bag_unknown


def train_bag_with_unknown():
    vocabulary = get_vocabulary()

    training_features, training_target = get_bag_unknown(vocabulary, 'training', max_size=12000)

    encoder = OneHotEncoder()
    encoder.fit(training_target)
    training_target = encoder.transform(training_target)

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


def validate_bag_with_unknown():
    vocabulary = get_vocabulary()

    validation_features, validation_target = get_bag_unknown(vocabulary, 'validation')

    encoder = OneHotEncoder()
    encoder.fit(validation_target)
    validation_target = encoder.transform(validation_target)

    with open('py_objects/sklearn_model.pickle', 'rb') as model_file:
        clf = pickle.load(model_file)
    validation_predict = clf.predict(validation_features)
    print(accuracy_score(validation_target, validation_predict))

    cm = confusion_matrix(validation_target.toarray().argmax(axis=1),
                          validation_predict.toarray().argmax(axis=1))
    plt.rcParams["pdf.use14corefonts"] = True
    plt.rcParams["ps.useafm"] = True

    sn.heatmap(cm, cmap="Reds", annot=True, cbar=False, fmt='n')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('charts/sklearn_validation_confusion_matrix.png', dpi=300)

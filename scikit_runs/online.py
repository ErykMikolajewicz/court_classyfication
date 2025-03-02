"""That file is a scratch, work but performance is very poor, probably using batches can help.
Accuracy is 78%, so here is also way for improvement.
It is an implementation of online learning with FeatureHasher, for better handling unknown words,
and to do not have to construct vocabulary."""
import pickle

from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.texts_corps import get_cases_words_count
from src.labeling import LABELS_REGEXES


def online_training():
    feature_hasher = FeatureHasher()
    counters = get_cases_words_count('training')

    le = LabelEncoder()
    labels = list(LABELS_REGEXES.values())
    labels_length = len(labels)
    le.fit(labels)
    clf = MLPClassifier(random_state=42, alpha=0.1, hidden_layer_sizes=(20, 20))
    for n, (counter, case_id) in enumerate(counters):
        features = feature_hasher.transform([counter]).toarray()
        target_index = le.transform([case_id])
        target = np.zeros((labels_length, 1))
        target[target_index] = 1
        clf.partial_fit(features, target.T, classes=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        print(n, case_id)

    with open('py_objects/sklearn_model.pickle', 'wb') as model_file:
        pickle.dump(clf, model_file)


def online_validate():
    feature_hasher = FeatureHasher()
    counters = get_cases_words_count('validation')

    le = LabelEncoder()
    labels = list(LABELS_REGEXES.values())
    labels_length = len(labels)
    le.fit(labels)

    with open('py_objects/sklearn_model.pickle', 'rb') as model_file:
        clf = pickle.load(model_file)
    n = 0
    for i, (counter, case_id) in enumerate(counters):
        features = feature_hasher.transform([counter]).toarray()
        target_index = le.transform([case_id])
        target = np.zeros((labels_length, 1))
        target[target_index] = 1
        result = clf.predict(features)
        if (result[0] == target[:, 0]).all():
            n += 1


    print('Wynik:', n/(i+1))

    with open('py_objects/sklearn_model.pickle', 'wb') as model_file:
        pickle.dump(clf, model_file)




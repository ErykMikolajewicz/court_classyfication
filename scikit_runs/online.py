"""That file is a scratch, work but performance is very poor, probably using batches can help.
Accuracy is 78%, so here is also way for improvement.
It is an implementation of online learning with FeatureHasher, for better handling unknown words,
and to do not have to construct vocabulary."""
import pickle
from typing import Literal

from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier

from src.texts_corps import get_cases_words_count
from src.labeling import get_labels


def online_training(label_type: Literal["detailed", "general"]):
    feature_hasher = FeatureHasher()
    counters = get_cases_words_count('training', label_type)

    labels = get_labels(label_type)
    one_hot_encoder = OneHotEncoder(categories=[labels])
    clf = MLPClassifier(random_state=42, alpha=0.1, hidden_layer_sizes=(20, 20))
    for n, (counter, case_id) in enumerate(counters):
        features = feature_hasher.transform([counter]).toarray()
        target = one_hot_encoder.transform([case_id])
        clf.partial_fit(features, target)
        print(n, case_id)

    with open('py_objects/sklearn_model.pickle', 'wb') as model_file:
        pickle.dump(clf, model_file)


def online_validate(label_type: Literal["detailed", "general"]):
    feature_hasher = FeatureHasher()
    counters = get_cases_words_count('validation', label_type)

    with open('py_objects/sklearn_model.pickle', 'rb') as model_file:
        clf = pickle.load(model_file)

    labels = get_labels(label_type)
    one_hot_encoder = OneHotEncoder(categories=[labels])
    n = 0
    for i, (counter, case_id) in enumerate(counters):
        features = feature_hasher.transform([counter]).toarray()
        target = one_hot_encoder.transform([case_id])
        result = clf.predict(features)
        if (result[0] == target[:, 0]).all():
            n += 1

    print('Wynik:', n/(i+1))

    with open('py_objects/sklearn_model.pickle', 'wb') as model_file:
        pickle.dump(clf, model_file)

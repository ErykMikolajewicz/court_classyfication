"""Using LDA make plot for bag of word features."""
from random import shuffle
from typing import Literal

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

from src.ml_preparing import get_bag_unknown
from src.texts_corps import get_vocabulary
from src.labeling import get_labels


def make_lda_plot(batch_size: int | None, label_type: Literal["detailed", "general"]):
    labels_names = get_labels(label_type)

    vocabulary = get_vocabulary()
    words_counts, labels = get_bag_unknown(vocabulary, 'training', label_type, max_size=batch_size)
    labels = labels.ravel()

    lda = LinearDiscriminantAnalysis(n_components=2)
    reduced_features = lda.fit_transform(words_counts, labels)

    n = len(labels_names)
    name='hsv'
    cmap = plt.get_cmap(name, n)

    markers = [".","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_", 1, 2, 3, 4, 5, 6, 7]
    shuffle(markers)

    for i, l in enumerate(labels_names):
        plt.scatter(reduced_features[labels==l, 0], reduced_features[labels==l, 1] * (-1), label= f'Class {l}',
                    c=cmap(i), marker=markers[i])

    plt.tight_layout(rect=(0, 0, 0.8, 1))
    plt.legend(labels_names, loc=(1.0, 0.5))

    plt.show()

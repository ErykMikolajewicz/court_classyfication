from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
import torch
from torchmetrics import Accuracy

from src.texts_corps import get_vocabulary
from src.ml_preparing import get_bag_unknown
from src.labeling import get_labels


torch.manual_seed(42)


def torch_train_bag_with_unknown(label_type: Literal["detailed", "general"]):
    vocabulary = get_vocabulary()

    training_features, training_target = get_bag_unknown(vocabulary, label_type, max_size=12_000, type_=np.float32)

    labels = get_labels(label_type)
    encoder = OneHotEncoder(categories=[labels], dtype=np.float32, sparse_output=False)
    training_target = encoder.fit_transform(training_target)

    input_parameters = training_features.shape[1]
    output_parameters = training_target.shape[1]

    training_features = torch.from_numpy(training_features)
    training_target = torch.from_numpy(training_target)

    model = nn.Sequential(
        nn.Linear(input_parameters, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, output_parameters),
        nn.Softmax()
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    num_epochs = 2_000
    accuracy = Accuracy(task='multiclass', num_classes=len(training_target[0]))

    def compute_accuracy(prediction_f) -> float:
        predictions = torch.argmax(prediction_f, dim=1)
        real_values = torch.argmax(training_target, dim=1)
        score_tensor = accuracy(predictions, real_values)
        score = score_tensor.item()
        return score

    loss_hist_train = [0] * num_epochs
    train_accuracy = [0] * num_epochs
    for epoch in range(num_epochs):
        print(epoch)
        prediction = model(training_features)[:, :]
        loss = loss_fn(prediction, training_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist_train[epoch] += loss.item()
        train_accuracy[epoch] = compute_accuracy(prediction)

    print(train_accuracy[-1])

    plt.plot(range(num_epochs), loss_hist_train)
    plt.plot(range(num_epochs), train_accuracy)
    plt.xlim(0, 2_000)
    plt.ylim(0, 1)
    plt.show()


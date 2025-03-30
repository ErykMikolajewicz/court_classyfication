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


def torch_train_bag_with_unknown2(label_type: Literal["detailed", "general"], batch_size=None):
    vocabulary = get_vocabulary()

    training_features, training_target = get_bag_unknown(vocabulary, label_type, max_size=12_000, type_=np.float32)

    labels = get_labels(label_type)
    encoder = OneHotEncoder(categories=[labels], dtype=np.float32, sparse_output=False)
    training_target = encoder.fit_transform(training_target)

    input_parameters = training_features.shape[1]
    output_parameters = training_target.shape[1]

    training_features = torch.from_numpy(training_features)
    training_target = torch.from_numpy(training_target)

    dataset = torch.utils.data.TensorDataset(training_features, training_target)

    if batch_size is None:
        batch_size = len(training_features)
        shuffle = False
    else:
        shuffle = True

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

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

    num_epochs = 200
    accuracy = Accuracy(task='multiclass', num_classes=len(training_target[0]))

    def compute_accuracy(prediction_f, target_f) -> float:
        predictions = torch.argmax(prediction_f, dim=1)
        real_values = torch.argmax(target_f, dim=1)
        score_tensor = accuracy(predictions, real_values)
        score = score_tensor.item()
        return score

    loss_hist_train = [0] * num_epochs
    train_accuracy = [0] * num_epochs

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        batches_per_epoch = 0

        for batch_features, batch_targets in dataloader:
            prediction = model(batch_features)
            loss = loss_fn(prediction, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += compute_accuracy(prediction, batch_targets)
            batches_per_epoch += 1

        loss_hist_train[epoch] = epoch_loss / batches_per_epoch
        train_accuracy[epoch] = epoch_accuracy / batches_per_epoch

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss_hist_train[epoch]:.4f}, Accuracy = {train_accuracy[epoch]:.4f}")

    print(f"Final training accuracy: {train_accuracy[-1]:.4f}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), loss_hist_train)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(0, num_epochs)

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracy)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(0, num_epochs)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

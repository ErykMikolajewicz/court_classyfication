from typing import Literal, Optional
import os

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchmetrics import Accuracy

from src.texts_corps import get_vocabulary
from src.ml_preparing import get_bag_unknown
from src.labeling import get_labels


torch.manual_seed(42)


def torch_train_bag_with_unknown(label_type: Literal["detailed", "general"], batch_size: Optional[int] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocabulary = get_vocabulary()

    training_features, training_target = get_bag_unknown(vocabulary, label_type, max_size=12_000, type_=np.float32)

    labels = get_labels(label_type)
    labels_indexes = {label: index for index, label in enumerate(labels)}
    training_target = np.array([labels_indexes[label.item()] for label in training_target], dtype=np.int64)

    input_parameters = training_features.shape[1]
    output_parameters = len(labels)

    training_features = torch.from_numpy(training_features)
    training_target = torch.from_numpy(training_target)

    dataset = torch.utils.data.TensorDataset(training_features, training_target)

    if batch_size is None:
        batch_size = len(training_features)
        shuffle = False
    else:
        shuffle = True

    cpu_number = os.cpu_count()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cpu_number)

    model = nn.Sequential(
        nn.Linear(input_parameters, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, output_parameters)
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    num_epochs = 200
    accuracy = Accuracy(task='multiclass', num_classes=len(labels)).to(device)

    loss_hist_train = [0] * num_epochs
    train_accuracy = [0] * num_epochs

    from datetime import datetime
    t1 = datetime.now()

    for epoch in range(num_epochs):
        epoch_loss = 0
        batches_per_epoch = 0

        for batch_features, batch_targets in dataloader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            prediction = model(batch_features)

            loss = loss_function(prediction, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            prediction_indexes = torch.argmax(prediction, dim=1)
            accuracy.update(prediction_indexes, batch_targets)
            batches_per_epoch += 1

        loss_hist_train[epoch] = epoch_loss / batches_per_epoch
        train_accuracy[epoch] = accuracy.compute().item()
        accuracy.reset()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss_hist_train[epoch]:.4f}, Accuracy = {train_accuracy[epoch]:.4f}")

    t2 = datetime.now()
    print(t2-t1)

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

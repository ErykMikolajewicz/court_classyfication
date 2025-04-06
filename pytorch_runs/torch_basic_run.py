from typing import Literal, Optional
import os

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split

from src.texts_corps import get_vocabulary
from src.ml_preparing import get_bag_unknown
from src.labeling import get_labels


torch.manual_seed(42)


def torch_train_bag_with_unknown(label_type: Literal['detailed', 'general', 'appeal'],
                                 batch_size: Optional[int] = None,
                                 max_size: Optional[int] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocabulary = get_vocabulary(document_frequency_threshold=50)
    print(len(vocabulary))
    training_features, training_target = get_bag_unknown(vocabulary, label_type, max_size=max_size, type_=np.float32)

    labels = get_labels(label_type)
    labels_indexes = {label: index for index, label in enumerate(labels)}
    training_target = np.array([labels_indexes[label.item()] for label in training_target], dtype=np.int64)

    training_features, validation_features, training_target, validation_target = train_test_split(
        training_features,
        training_target,
        test_size=0.3,
        random_state=42,
        stratify=training_target
    )

    input_parameters = training_features.shape[1]
    output_parameters = len(labels)

    training_features = torch.from_numpy(training_features)
    training_target = torch.from_numpy(training_target)

    training_dataset = torch.utils.data.TensorDataset(training_features, training_target)

    validation_features = torch.from_numpy(validation_features)
    validation_target = torch.from_numpy(validation_target)
    validation_dataset = torch.utils.data.TensorDataset(validation_features, validation_target)

    if batch_size is None:
        batch_size = len(training_features)
        shuffle = False
    else:
        shuffle = True

    cpu_number = os.cpu_count()
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=cpu_number)

    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=cpu_number)

    model = nn.Sequential(
        nn.Linear(input_parameters, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, output_parameters)
    ).to(device)

    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()
    l1_lambda = 0.001

    accuracy = Accuracy(task='multiclass', num_classes=len(labels)).to(device)

    num_epochs = 100

    loss_hist_train = [0] * num_epochs
    train_accuracy = [0] * num_epochs

    val_accuracy = [0] * num_epochs
    val_loss_hist = [0] * num_epochs

    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0
        batches_per_epoch = 0

        for t_batch_features, t_batch_targets in training_dataloader:
            batch_features, batch_targets = t_batch_features.to(device), t_batch_targets.to(device)
            prediction = model(batch_features)

            loss = loss_function(prediction, batch_targets)

            l1_norm = sum(param.abs().sum() for param in model.parameters())
            loss = loss + l1_lambda * l1_norm

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

        model.eval()
        validation_loss = 0
        validation_batches = 0

        with torch.no_grad():
            for batch_features, batch_targets in validation_dataloader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                batch_predictions = model(batch_features)

                loss = loss_function(batch_predictions, batch_targets)
                validation_loss += loss.item()
                validation_batches += 1

                predictions_indexes = torch.argmax(batch_predictions, dim=1)
                accuracy.update(predictions_indexes, batch_targets)

        val_loss_hist[epoch] = validation_loss / validation_batches
        val_accuracy[epoch] = accuracy.compute().item()
        accuracy.reset()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss_hist_train[epoch]:.4f} | Train Acc: {train_accuracy[epoch]:.4f}"
                  f" | Val Loss: {val_loss_hist[epoch]:.4f} | Val Acc: {val_accuracy[epoch]:.4f}")

    print(f"Final training accuracy: {train_accuracy[-1]:.4f}")

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.plot(range(num_epochs), loss_hist_train)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(0, num_epochs)

    plt.subplot(2, 2, 2)
    plt.plot(range(num_epochs), train_accuracy)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(0, num_epochs)
    plt.ylim(0, 1)

    plt.subplot(2, 2, 3)
    plt.plot(range(num_epochs), val_loss_hist)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(0, num_epochs)

    plt.subplot(2, 2, 4)
    plt.plot(range(num_epochs), val_accuracy)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(0, num_epochs)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

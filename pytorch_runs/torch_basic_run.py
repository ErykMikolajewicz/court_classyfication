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


class BagTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocabulary = get_vocabulary(document_frequency_threshold=50)
        self.vocabulary_size = len(self.vocabulary)

        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.__training_loader = None
        self.__validation_loader = None
        self.labels = None
        self.__accuracy_metric = None
        self.train_loss_hist = []
        self.train_accuracy_hist = []
        self.val_loss_hist = []
        self.val_accuracy_hist = []

    def prepare_data(self,
                     label_type: Literal['detailed', 'general', 'appeal'],
                     max_size: Optional[int] = None,
                     batch_size: Optional[int] = None):
        features, targets = get_bag_unknown(self.vocabulary, label_type, max_size=max_size, type_=np.float32)
        self.labels = get_labels(label_type)
        self.__accuracy_metric = Accuracy(task='multiclass', num_classes=len(self.labels)).to(self.device)
        label_to_index = {label: i for i, label in enumerate(self.labels)}
        targets = np.array([label_to_index[label.item()] for label in targets], dtype=np.int64)

        training_features, validation_features, training_target, validation_target = train_test_split(
            features, targets, test_size=0.3, random_state=42, stratify=targets)

        training_features, validation_features = map(torch.from_numpy, [training_features, validation_features])
        training_target, validation_target = map(torch.from_numpy, [training_target, validation_target])

        train_dataset = torch.utils.data.TensorDataset(training_features, training_target)
        val_dataset = torch.utils.data.TensorDataset(validation_features, validation_target)

        if batch_size is None:
            batch_size = len(train_dataset)
            shuffle = False
        else:
            shuffle = True

        num_workers = os.cpu_count()
        self.__training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,
                                                             shuffle=shuffle, num_workers=num_workers)
        self.__validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2048, pin_memory=True,
                                                               num_workers=num_workers)

    def train(self,
              num_epochs: int = 100,
              l1_lambda: float = 0.001,
              initial_lr: float = 0.01):
        self.train_loss_hist = []
        self.train_accuracy_hist = []
        self.val_loss_hist = []
        self.val_accuracy_hist = []

        cross_entropy_loss = nn.CrossEntropyLoss()
        def loss_function(outputs, targets):
            l1_norm = sum(param.abs().sum() for param in self.model.parameters())
            return cross_entropy_loss(outputs, targets) + l1_lambda * l1_norm
        self.loss_function = loss_function

        input_dim = self.vocabulary_size + 1 # + 1 is for unknown tokens
        output_dim = len(self.labels)

        self.model = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, output_dim)
        ).to(self.device)

        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=initial_lr)

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss, batch_counter = 0, 0

            for features, targets in self.__training_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)
                loss = self.loss_function(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                self.__accuracy_metric.update(predictions, targets)
                batch_counter += 1

            self.train_loss_hist.append(epoch_loss / batch_counter)
            self.train_accuracy_hist.append(self.__accuracy_metric.compute().item())
            self.__accuracy_metric.reset()

            val_loss = self.evaluate()

            if epoch % 5 == 0:
                print(f"Epoch {epoch} | Train Loss: {self.train_loss_hist[-1]:.4f} | "
                      f"Train Acc: {self.train_accuracy_hist[-1]:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {self.val_accuracy_hist[-1]:.4f}")

    def evaluate(self):
        self.model.eval()
        batches_loss, batch_number = 0, 0

        with torch.no_grad():
            for features, targets in self.__validation_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)

                loss = self.loss_function(outputs, targets)
                batches_loss += loss.item()
                batch_number += 1

                predictions = torch.argmax(outputs, dim=1)
                self.__accuracy_metric.update(predictions, targets)

        validation_loss = batches_loss / batch_number
        self.val_loss_hist.append(validation_loss)
        self.val_accuracy_hist.append(self.__accuracy_metric.compute().item())
        self.__accuracy_metric.reset()
        return validation_loss

    def plot_metrics(self):

        plt.figure(figsize=(10, 5))
        plt.subplot(2, 2, 1)
        plt.plot(range(len(self.train_loss_hist)), self.train_loss_hist)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(2, 2, 2)
        plt.plot(range(len(self.train_accuracy_hist)), self.train_accuracy_hist)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)

        plt.subplot(2, 2, 3)
        plt.plot(range(len(self.val_loss_hist)), self.val_loss_hist)
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(2, 2, 4)
        plt.plot(range(len(self.val_accuracy_hist)), self.val_accuracy_hist)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.show()
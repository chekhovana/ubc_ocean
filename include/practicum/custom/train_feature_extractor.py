import os
import datetime
import copy
import re
import yaml
import uuid
import warnings
import time
import inspect

import numpy as np
import pandas as pd
from functools import partial, reduce
from random import shuffle
import random

import torch
from torch import nn, optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchvision.models import resnet
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, \
    accuracy_score
from sklearn import metrics as mtx
from sklearn import model_selection as ms


class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2],
                                          num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                     padding=(3, 3), bias=False)

    def forward(self, x):
        return torch.softmax(super(MnistResNet, self).forward(x), dim=-1)


def get_data_loaders(train_batch_size, val_batch_size):
    root = '../data'
    mnist = MNIST(download=True, train=True, root=root).data.float()

    data_transform = Compose([Resize((224, 224)), ToTensor(),
                              Normalize((mnist.mean() / 255,),
                                        (mnist.std() / 255,))])

    train_loader = DataLoader(
        MNIST(download=True, root=f'{root}', transform=data_transform,
              train=True),
        batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(
        MNIST(download=False, root=f'{root}', transform=data_transform,
              train=False),
        batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def calculate_metric(metric_fn, true_y, pred_y):
    # multi class problems need to have averaging method
    # if "average" in inspect.getfullargspec(metric_fn).args:
    try:
        return metric_fn(true_y, pred_y, average="macro")
    except:
        return metric_fn(true_y, pred_y)


def print_scores(p, r, f1, a, batch_size):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"),
                            (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores) / batch_size:.4f}")


def main():
    train_batch_size = 256
    val_batch_size = 256
    train_loader, valid_loader = get_data_loaders(train_batch_size,
                                                  val_batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MnistResNet().to(device)

    epochs = 5
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())

    losses = []
    batches = len(train_loader)
    val_batches = len(valid_loader)

    for epoch in range(epochs):
        total_loss = 0
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
        model.train()

        for i, data in progress:
            X, y = data[0].to(device), data[1].to(device)
            # training step for single batch
            model.zero_grad()  # to make sure that all the grads are 0
            outputs = model(X)  # forward
            loss = loss_function(outputs, y)  # get loss
            loss.backward()  # accumulates the gradient (by addition) for each parameter.
            optimizer.step()  # performs a parameter update based on the current gradient

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

            # updating progress bar
            progress.set_description(
                "Loss: {:.4f}".format(total_loss / (i + 1)))

        # releasing unceseccary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----------------- VALIDATION  -----------------
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []

        # set model to evaluating (testing)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                X, y = data[0].to(device), data[1].to(device)

                outputs = model(X)  # this get's the prediction from the network

                val_losses += loss_function(outputs, y)

                predicted_classes = torch.max(outputs, 1)[
                    1]  # get class from network's prediction

                # calculate P/R/F1/A metrics for batch
                for acc, metric in zip((precision, recall, f1, accuracy),
                                       (precision_score, recall_score, f1_score,
                                        accuracy_score)):
                    acc.append(
                        calculate_metric(metric, y.cpu(),
                                         predicted_classes.cpu())
                    )

        print(
            f"Epoch {epoch + 1}/{epochs}, training loss: {total_loss / batches}, validation loss: {val_losses / val_batches}")
        print_scores(precision, recall, f1, accuracy, val_batches)
        losses.append(total_loss / batches)  # for plotting learning curve
        torch.save(model.state_dict(), 'mnist_state.pt')
        # %% md


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from itertools import chain
from logging import getLogger
from time import time

import numpy as np
import torch
from frozendict import frozendict
from sklearn.utils.multiclass import type_of_target
from torch import nn
from torch.nn.functional import binary_cross_entropy, cross_entropy, mse_loss

logger = getLogger(__name__)


class EntityEmbeddingNN(nn.Module):
    def __init__(
            self,
            n_uniques: np.ndarray,
            A=10, B=5,
            dropout1=0.1,
            dropout2=0.1,
            n_class=2
    ):
        super(EntityEmbeddingNN, self).__init__()
        self.epoch = 0
        self.n_class = n_class
        self.dropout2 = dropout2
        self.dropout1 = dropout1
        self.n_uniques = n_uniques
        self.A = A
        self.B = B
        exp_ = np.exp(-n_uniques * 0.05)
        self.embed_dims = np.round(5 * (1 - exp_) + 1).astype("int")
        sum_ = np.log(self.embed_dims).sum()
        self.n_layer1 = min(1000,
                            int(A * (n_uniques.size ** 0.5) * sum_ + 1))
        self.n_layer2 = int(self.n_layer1 / B) + 2
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(n_unique), int(embed_dim))
            for n_unique, embed_dim in zip(self.n_uniques, self.embed_dims)
        ])
        self.layer1 = nn.Sequential(
            nn.Linear(self.embed_dims.sum(), self.n_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout1)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(self.n_layer1, self.n_layer2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout2)
        )
        self.dense = nn.Sequential(
            self.layer1,
            self.layer2,
        )
        # regression
        if n_class == 1:
            self.output = nn.Linear(self.n_layer2, 1)
        # binary classification
        elif n_class == 2:
            self.output = nn.Sequential(
                nn.Linear(self.n_layer2, 1),
                nn.Sigmoid()
            )
        # multi classification
        elif n_class > 2:
            self.output = nn.Sequential(
                nn.Linear(self.n_layer2, n_class),
                nn.Softmax()
            )
        else:
            raise ValueError(f"Invalid n_class : {n_class}")
        for m in chain(self.dense.modules(), self.output.modules(), self.embeddings.modules()):
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, X: np.ndarray):
        embeds = [self.embeddings[i](torch.from_numpy(X[:, i].astype("int64")))
                  for i in range(X.shape[1])]
        features = self.dense(torch.cat(embeds, dim=1))
        outputs = self.output(features)
        return embeds, features, outputs


def train_entity_embedding_nn(
        X: np.ndarray,
        y: np.ndarray,
        lr=1e-2, epoch=25,
        init_model=None,
        callback=None,
        n_class=None,
        nn_params=frozendict()
) -> EntityEmbeddingNN:
    # fixme: tricky operate
    n_uniques = (X.max(axis=0) + 1).astype("int")
    if n_class is None:
        if type_of_target(y.astype("float")) == "continuous":
            n_class = 1
        else:
            n_class = np.unique(y).size
    nn_params = dict(nn_params)
    nn_params.update(n_class=n_class)
    if init_model is None:
        entity_embedding_nn: nn.Module = EntityEmbeddingNN(
            n_uniques, **nn_params
        )
    else:
        entity_embedding_nn = init_model
    entity_embedding_nn.train(True)
    optimizer = torch.optim.Adam(entity_embedding_nn.parameters(), lr=lr)

    start = time()
    if n_class > 2:
        y_tensor = torch.from_numpy(y).long()
    else:
        y_tensor = torch.from_numpy(y).double()
    init_epoch = getattr(entity_embedding_nn, "epoch", 0)
    for i in range(init_epoch, epoch):
        optimizer.zero_grad()
        _, _, outputs = entity_embedding_nn(X)
        if n_class == 2:
            loss = binary_cross_entropy(outputs.flatten().double(), y_tensor)
        elif n_class > 2:
            loss = cross_entropy(outputs.double(), y_tensor)
        elif n_class == 1:
            loss = mse_loss(outputs.flatten().double(), y_tensor)
        else:
            raise ValueError
        loss.backward()
        optimizer.step()
        if callback is not None:
            callback(i, entity_embedding_nn)
        entity_embedding_nn.epoch = i
    end = time()
    logger.info(f"EntityEmbeddingNN training time = {end - start:.2f}s")
    entity_embedding_nn.eval()
    return entity_embedding_nn

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from math import ceil
from time import time
from typing import List, Union, Optional

import numpy as np
import torch
from frozendict import frozendict
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from torch import nn
from torch.nn.functional import cross_entropy, mse_loss

from autoflow.utils.data import check_n_jobs
from autoflow.utils.logging_ import get_logger


class TabularNN(nn.Module):
    def __init__(
            self,
            n_uniques: np.ndarray,
            vector_dim: int,
            cat_indexes: Union[List[int], np.ndarray],
            max_layer_width=2056,
            min_layer_width=32,
            dropout_hidden=0.1,
            af_hidden="relu",
            af_output="linear",
            dropout_output=0.2,
            layers=(256, 128),
            n_class=2,
            use_bn=True
    ):
        super(TabularNN, self).__init__()
        self.logger = get_logger(__name__)
        self.af_output = af_output
        self.af_hidden = af_hidden
        self.max_epoch = 0
        self.use_bn = use_bn
        assert len(cat_indexes) == len(n_uniques)
        self.layers = layers
        self.min_layer_width = min_layer_width
        self.max_layer_width = max_layer_width
        self.cat_indexes = np.array(cat_indexes, dtype="int")
        self.n_class = n_class
        self.dropout_output = dropout_output
        self.dropout_hidden = dropout_hidden
        self.n_uniques = n_uniques
        num_features = len(n_uniques) + vector_dim
        prop_vector_features = vector_dim / num_features
        msg = ""
        if vector_dim > 0:
            numeric_embed_dim = int(np.clip(
                round(layers[0] * prop_vector_features * np.log10(vector_dim + 10)),
                min_layer_width, max_layer_width
            ))
            msg += f"numeric_embed_dim = {numeric_embed_dim}; "
            self.numeric_block = nn.Sequential(
                nn.Linear(vector_dim, numeric_embed_dim),
                self.get_activate_function(self.af_hidden)
            )
        else:
            numeric_embed_dim = 0
            msg += f"numeric_block is None; "
            self.numeric_block = None
        if len(n_uniques) > 0:
            exp_ = np.exp(-n_uniques * 0.05)
            self.embed_dims = np.round(5 * (1 - exp_) + 1).astype("int")
            self.embedding_blocks = nn.ModuleList([
                nn.Embedding(int(n_unique), int(embed_dim))
                for n_unique, embed_dim in zip(self.n_uniques, self.embed_dims)
            ])
            msg += f"embed_dims.sum() = {self.embed_dims.sum()}; "
        else:
            msg += f"embedding_blocks is None; "
            self.embed_dims = np.array([])
            self.embedding_blocks = None
        after_embed_dim = int(self.embed_dims.sum() + numeric_embed_dim)
        deep_net_modules = []
        layers_ = [after_embed_dim] + list(layers)
        layers_len = len(layers_)
        for i in range(1, layers_len):
            in_features = layers_[i - 1]
            out_features = layers_[i]
            msg += f"layer{i} = {in_features}->{out_features}; "
            dropout_rate = self.dropout_hidden
            block = self.get_block(
                in_features, out_features,
                use_bn=self.use_bn, dropout_rate=dropout_rate, af_name=self.af_hidden)
            deep_net_modules.append(block)
        self.logger.info(msg)
        deep_net_modules.append(
            self.get_block(
                layers_[-1], self.n_class,
                self.use_bn, dropout_rate=self.dropout_output, af_name=self.af_output
            ))
        self.deep_net = nn.Sequential(*deep_net_modules)
        self.wide_net = self.get_block(
            after_embed_dim, n_class,
            use_bn=self.use_bn, dropout_rate=self.dropout_output, af_name=self.af_output
        )
        output_modules = []
        if self.n_class > 1:
            output_modules.append(nn.Softmax(dim=1))
        self.output_layer = nn.Sequential(*output_modules)
        modules = [
            self.deep_net.modules(),
            self.wide_net.modules(),
            self.output_layer.modules(),
        ]
        if self.embedding_blocks is not None:
            modules.append(self.embedding_blocks)
        if self.numeric_block is not None:
            modules.append(self.numeric_block)
        for m in modules:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                m.bias.data.zero_()

    def get_activate_function(self, af_name: str):
        af_name = af_name.lower()
        if af_name == "relu":
            return nn.ReLU(inplace=True)
        elif af_name == "leaky_relu":
            return nn.LeakyReLU(inplace=True)
        elif af_name == "elu":
            return nn.ELU(inplace=True)
        elif af_name == "linear":
            return nn.Identity()
        elif af_name == "tanh":
            return nn.Tanh()
        elif af_name == "sigmoid":
            return nn.Sigmoid()
        elif af_name == "softplus":
            return nn.Softplus()
        else:
            raise ValueError(f"Unknown activate function name {af_name}")

    def get_block(self, in_features, out_features, use_bn, dropout_rate, af_name):
        seq = []
        seq.append(nn.Linear(in_features, out_features))
        if use_bn:
            seq.append(nn.BatchNorm1d(out_features))
        seq.append(self.get_activate_function(af_name))
        if dropout_rate > 0:
            seq.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*seq)

    def forward(self, X: np.ndarray):
        embeds = []
        if self.embedding_blocks is not None:
            embeds += [self.embedding_blocks[i](torch.from_numpy(X[:, col].astype("int64")))
                       for i, col in enumerate(self.cat_indexes)]
        num_indexed = np.setdiff1d(np.arange(X.shape[1]), self.cat_indexes)
        if self.numeric_block is not None:
            embeds.append(self.numeric_block(torch.from_numpy(X[:, num_indexed].astype("float32"))))
        cat_embeds = torch.cat(embeds, dim=1)
        outputs = self.deep_net(cat_embeds) + self.wide_net(cat_embeds)
        activated = self.output_layer(outputs)
        return activated


def train_tabular_nn(
        X: np.ndarray,
        y: np.ndarray,
        cat_indexes,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        lr=1e-2,
        max_epoch=25,
        init_model=None,
        callback=None,
        n_class=None,
        nn_params=frozendict(),
        random_state=1000,
        batch_size=2048,
        optimizer="adam",
        n_jobs=-1,
        class_weight=None
) -> TabularNN:
    n_jobs = check_n_jobs(n_jobs)
    rng = check_random_state(random_state)
    torch.manual_seed(rng.randint(0, 10000))
    torch.set_num_threads(n_jobs)
    # np.random.seed(random_state)
    cat_indexes = np.array(cat_indexes, dtype="int")
    n_uniques = (X[:, cat_indexes].max(axis=0) + 1).astype("int")
    vector_dim = X.shape[1] - len(cat_indexes)
    if n_class is None:
        if type_of_target(y.astype("float")) == "continuous":
            n_class = 1
        else:
            n_class = np.unique(y).size
    nn_params = dict(nn_params)
    nn_params.update(n_class=n_class)
    if init_model is None:
        tabular_nn: nn.Module = TabularNN(
            n_uniques, vector_dim, cat_indexes,
            **nn_params
        )
    else:
        tabular_nn = init_model
    if optimizer == "adam":
        nn_optimizer = torch.optim.Adam(tabular_nn.parameters(), lr=lr)
    elif optimizer == "sgd":
        nn_optimizer = torch.optim.SGD(tabular_nn.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer {optimizer}")
    start = time()
    if n_class >= 2:
        y_tensor = torch.from_numpy(y).long()
    else:
        y_tensor = torch.from_numpy(y).double()
    if n_class >= 2 and class_weight == "balanced":
        unique, counts = np.unique(y, return_counts=True)
        counts = counts / np.min(counts)
        weight = torch.from_numpy(1 / counts).double()
    else:
        weight = None
    init_epoch = getattr(tabular_nn, "max_epoch", 0)
    for epoch_index in range(init_epoch, max_epoch):
        # todo : batch(OK) validate(OK)  warm_start(OK)
        # todo : early_stopping(OK) multiclass_metric(OK) sample_weight
        tabular_nn.train(True)
        # batch
        permutation = rng.permutation(len(y))
        batch_ixs = [permutation[i * batch_size:(i + 1) * batch_size] for i in range(ceil(len(y) / batch_size))]
        for batch_ix in batch_ixs:
            nn_optimizer.zero_grad()
            outputs = tabular_nn(X[batch_ix, :])
            if n_class >= 2:
                loss = cross_entropy(outputs.double(), y_tensor[batch_ix], weight=weight)
            elif n_class == 1:
                loss = mse_loss(outputs.flatten().double(), y_tensor[batch_ix])
            else:
                raise ValueError
            loss.backward()
            nn_optimizer.step()
        if callback is not None:
            if callback(epoch_index, tabular_nn, X, y, X_valid, y_valid) == True:
                break
    end = time()
    tabular_nn.max_epoch = max_epoch
    tabular_nn.logger.info(f"TabularNN training time = {end - start:.2f}s")
    tabular_nn.eval()
    return tabular_nn

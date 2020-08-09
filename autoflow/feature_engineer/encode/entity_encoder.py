#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import functools
import warnings
from copy import copy

import category_encoders.utils as util
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils._random import check_random_state
from sklearn.utils.multiclass import type_of_target

from autoflow.tnn.entity_embedding_nn import TrainEntityEmbeddingNN, EntityEmbeddingNN
from autoflow.utils.logging_ import get_logger
warnings.simplefilter(action='ignore', category=FutureWarning)

class EntityEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            cols=None,
            return_df=True,
            lr=1e-2,
            max_epoch=25,
            A=10,
            B=5,
            dropout1=0.1,
            dropout2=0.1,
            random_state=1000,
            verbose=1,
            n_jobs=-1,
            class_weight=None,
            batch_size=1024,
            optimizer="adam",
            normalize=True,
            copy=True
    ):
        self.normalize = normalize
        self.copy = copy
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.lr = lr
        self.dropout2 = dropout2
        self.dropout1 = dropout1
        self.B = B
        self.A = A
        self.max_epoch = max_epoch
        self.return_df = return_df
        self.drop_cols = []
        self.cols = cols
        self._dim = None
        self.feature_names = None
        self.model = None
        self.logger = get_logger(self)
        self.nn_params = {
            "A": self.A,
            "B": self.B,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2,
        }
        self.rng = check_random_state(self.random_state)
        self.trainer = TrainEntityEmbeddingNN(
            lr=self.lr,
            max_epoch=self.max_epoch,
            n_class=None,
            nn_params=self.nn_params,
            random_state=self.rng,
            batch_size=batch_size,
            optimizer=optimizer,
            n_jobs=self.n_jobs,
            class_weight=class_weight
        )
        self.scaler = StandardScaler(copy=True)
        self.keep_going = False

    def fit(self, X, y=None, **kwargs):
        # first check the type
        X = util.convert_input(X)
        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)
        if len(self.cols) == 0:
            self.keep_going = True
            return self
        self.ordinal_encoder = OrdinalEncoder(dtype=np.int)
        # 1. use sklearn's OrdinalEncoder convert categories to int
        self.ordinal_encoder.fit(X[self.cols])
        X_ordinal = self.ordinal_encoder.transform(X[self.cols])
        n_uniques = X_ordinal.max(axis=0).astype("int") + 1
        if self.normalize and type_of_target(y) != "continuous":
            self.normalize = False
        if self.normalize:
            y = self.scaler.fit_transform(y[:, None]).flatten()
        # 2. train_entity_embedding_nn
        self.model = self.trainer.train(
            self.model, EntityEmbeddingNN, X_ordinal, y, None, None,
            self.callback, n_uniques=n_uniques
        )
        return self

    def transform(self, X):
        if self.keep_going:
            return X
        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')

        # first check the type
        X = util.convert_input(X)
        if self.copy:
            X = copy(X)
        index = X.index
        X.index = range(X.shape[0])
        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        if not self.cols:
            return X if self.return_df else X.values

        # 1. convert X to X_ordinal, and handle unknown categories
        is_known_categories = []
        for i, col in enumerate(self.cols):
            categories = self.ordinal_encoder.categories_[i]
            is_known_category = X[col].isin(categories).values
            if not np.all(is_known_category):
                X.loc[~is_known_category, col] = categories[0]
            is_known_categories.append(is_known_category)
        X_ordinal = self.ordinal_encoder.transform(X[self.cols])
        # 2. embedding by nn, and handle unknown categories by fill 0
        X_embeds, _ = self.model(X_ordinal)
        X_embeds = [X_embed.detach().numpy() for X_embed in X_embeds]
        for i, is_known_category in enumerate(is_known_categories):
            if not np.all(is_known_category):
                X_embeds[i][~is_known_category, :] = 0
        # 3. replace origin
        get_valid_col_name = functools.partial(self.get_valid_col_name, df=X)
        col2idx = dict(zip(self.cols, range(len(self.cols))))
        result_df_list = []
        cur_columns = []
        for column in X.columns:
            if column in self.cols:
                if len(cur_columns) > 0:
                    result_df_list.append(X[cur_columns])
                    cur_columns = []
                idx = col2idx[column]
                embed = X_embeds[idx]
                new_columns = [f"{column}_{i}" for i in range(embed.shape[1])]
                new_columns = [get_valid_col_name(new_column) for new_column in
                               new_columns]  # fixme Maybe it still exists bug
                embed = pd.DataFrame(embed, columns=new_columns)
                result_df_list.append(embed)
            else:
                cur_columns.append(column)
        if len(cur_columns) > 0:
            result_df_list.append(X[cur_columns])
            cur_columns = []
        X = pd.concat(result_df_list, axis=1)
        X.index = index
        if self.return_df:
            return X
        else:
            return X.values

    def get_valid_col_name(self, col_name, df: pd.DataFrame):
        while col_name in df.columns:
            col_name += "_"
        return col_name

    def callback(self, epoch_index, model, X, y, X_valid, y_valid) -> bool:
        model.eval()
        should_print = self.verbose > 0 and epoch_index % self.verbose == 0
        n_class = getattr(model, "n_class", 1)
        if n_class == 1:
            score_func = r2_score
        else:
            score_func = accuracy_score
        score_func_name = score_func.__name__
        _, y_pred = model(X)
        y_pred = y_pred.detach().numpy()
        if n_class > 1:
            y_pred = y_pred.argmax(axis=1)
        # if self.normalize:
        #     y_pred=self.scaler.inverse_transform(y_pred[:,None]).flatten()
        train_score = score_func(y, y_pred)
        msg = f"epoch_index = {epoch_index}, " \
            f"TrainSet {score_func_name} = {train_score:.3f}"
        if should_print:
            self.logger.info(msg)
        else:
            self.logger.debug(msg)
        return False

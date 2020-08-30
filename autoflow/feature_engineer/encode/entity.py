#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import functools
import warnings
from copy import copy, deepcopy
from time import time

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
            copy=True,
            budget=10,
            early_stopping_rounds=5,
            # warm_start=False,
            # accepted_samples=30,
            update_epoch=5,
            update_accepted_samples=10,
            update_used_samples=100,
    ):
        self.update_used_samples = update_used_samples
        self.update_epoch = update_epoch
        self.update_accepted_samples = update_accepted_samples
        # self.accepted_samples = accepted_samples
        # self.warm_start = warm_start
        self.early_stopping_rounds = early_stopping_rounds
        self.budget = budget
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
        self.rng: np.random.RandomState = check_random_state(self.random_state)
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
        self.iter = 0
        self.is_classification = None
        self.samples_db = [pd.DataFrame(), np.array([])]
        self.final_observations = self.get_initial_final_observations()
        self.n_uniques = None
        self.transform_matrix = None
        self.stage = ""
        self.refit_times = 0

    def get_initial_final_observations(self):
        return [pd.DataFrame(), np.array([])]

    def init_variables(self):
        self.learning_curve = [
            [],  # train_sizes_abs [0]
            [],  # train_scores    [1]
        ]
        self.performance_history = np.full(self.early_stopping_rounds, -np.inf)
        self.iteration_history = np.full(self.early_stopping_rounds, 0, dtype="int32")
        N = len(self.performance_history)
        self.best_estimators = np.zeros([N], dtype="object")
        self.early_stopped = False
        self.best_iteration = 0

    def initial_fit(self, X: pd.DataFrame, y: np.ndarray):
        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)
        self.ordinal_encoder = OrdinalEncoder(dtype=np.int)
        # 1. use sklearn's OrdinalEncoder convert categories to int
        self.ordinal_encoder.fit(X[self.cols])
        self.is_classification = (type_of_target(y) != "continuous")
        if self.normalize and self.is_classification:
            self.normalize = False
        if self.normalize:
            self.scaler.fit(y[:, None])

    def _fit(self, X: pd.DataFrame, y: np.ndarray):
        X_ordinal = self.ordinal_encoder.transform(X[self.cols])
        if self.n_uniques is None:
            self.n_uniques = X_ordinal.max(axis=0).astype("int") + 1
        if self.normalize:
            y = self.scaler.transform(y[:, None]).flatten()
        # 2. train_entity_embedding_nn
        self.model = self.trainer.train(
            self.model, EntityEmbeddingNN, X_ordinal, y, None, None,
            self.callback, n_uniques=self.n_uniques
        )

    def fit(self, X, y=None, **kwargs):
        self.init_variables()
        # first check the type
        X = util.convert_input(X)
        # fixme : 默认是warm start的
        self._dim = X.shape[1]
        # todo add logging_level, verbose
        if self.samples_db[0].shape[0] == 0:
            self.logger.info('Initial fitting')
            # todo handle n_choices <=3 by using equidistence_encoder
            self.initial_fit(X, y)
        if len(self.cols) == 0:
            self.keep_going = True
            return self
        self.start_time = time()
        if self.samples_db[0].shape[0] == 0:
            self.trainer.max_epoch = self.max_epoch
            self._fit(X, y)
            self.samples_db[0] = pd.concat([self.samples_db[0], X], axis=0).reset_index(drop=True)
            self.samples_db[1] = np.hstack([self.samples_db[1], y])
            self.transform_matrix = self.get_transform_matrix()
            self.stage = "Initial fitting"
            # todo early_stopping choose best model
        else:
            self.model.max_epoch = 0
            self.trainer.max_epoch = self.update_epoch
            # update final_observations
            self.final_observations[0] = pd.concat([self.final_observations[0], X], axis=0).reset_index(drop=True)
            self.final_observations[1] = np.hstack([self.final_observations[1], y])
            observations = self.final_observations[0].shape[0]
            if observations < self.update_accepted_samples:
                self.logger.info(f"only have {observations} observations, didnt training model.")
                self.transform_matrix = None
            else:
                n_used_samples = min(self.update_used_samples - observations, self.samples_db[0].shape[0])
                indexes = self.rng.choice(np.arange(self.samples_db[0].shape[0]), n_used_samples, False)
                # origin samples_db + final_observations -> X, y
                X_ = pd.concat([self.samples_db[0].loc[indexes, :], self.final_observations[0]]).reset_index(drop=True)
                y_ = np.hstack([self.samples_db[1][indexes], self.final_observations[1]])
                # fitting (using previous model)
                self._fit(X_, y_)
                # update samples_db by final_observations
                self.samples_db[0] = pd.concat([self.samples_db[0], self.final_observations[0]], axis=0). \
                    reset_index(drop=True)
                self.samples_db[1] = np.hstack([self.samples_db[1], self.final_observations[1]])
                # clear final_observations
                self.final_observations = self.get_initial_final_observations()
                self.transform_matrix = self.get_transform_matrix()
                self.refit_times += 1
                self.stage = f"refit-{self.refit_times}-times"
        return self

    def get_transform_matrix(self):
        # todo: 测试多个离散变量字段的情况
        N = self.n_uniques.max()
        M = self.n_uniques.size
        X_ordinal = np.zeros([N, M])
        for i, n_unique in enumerate(self.n_uniques):
            X_ordinal[:, i][:n_unique] = np.arange(n_unique)
        X_embeds, _ = self.model(X_ordinal)
        X_embeds = [X_embed.detach().numpy() for X_embed in X_embeds]
        for i, n_unique in enumerate(self.n_uniques):
            X_embeds[i] = X_embeds[i][:n_unique, :]
        return X_embeds

    def transform(self, X):
        if self.keep_going:
            return X
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

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
        self.iter = epoch_index
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
        if time() - self.start_time > self.budget:
            self.logger.info(f"Exceeded budget time({self.budget}), {self.__class__.__name__} is early stopping ...")
            return True
        if np.any(train_score > self.performance_history):
            index = epoch_index % self.early_stopping_rounds
            self.best_estimators[index] = deepcopy(model)
            self.performance_history[index] = train_score
            self.iteration_history[index] = epoch_index
        else:
            self.early_stopped = True
            self.logger.info(f"performance in training set no longer increase "
                             f"in {self.early_stopping_rounds} times, early stopping ...")
        return self.early_stopped

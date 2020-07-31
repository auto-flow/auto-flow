#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_array

from autoflow.nn.tabular_nn import train_tabular_nn
from autoflow.utils.logging_ import get_logger


class TabularNNEstimator(BaseEstimator):
    is_classification = None

    # todo: constraint cpu usage
    def __init__(
            self,
            max_layer_width=2056,
            min_layer_width=32,
            dropout_hidden=0.1,
            af_hidden="relu",
            af_output="linear",
            dropout_output=0.2,
            layer1=256,
            layer2=128,
            use_bn=True,
            lr=1e-2,
            max_epoch=32,
            random_state=1000,
            batch_size=1024,
            optimizer="adam",
            early_stopping_rounds=8,
            early_stopping_tol=0,
            verbose=-1,
            n_jobs=-1,
            class_weight=None
    ):
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.early_stopping_tol = early_stopping_tol
        self.early_stopping_rounds = early_stopping_rounds
        assert self.is_classification is not None, NotImplementedError
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.random_state = random_state
        self.max_epoch = max_epoch
        self.lr = lr
        self.use_bn = use_bn
        self.layer2 = layer2
        self.layer1 = layer1
        self.dropout_output = dropout_output
        self.af_output = af_output
        self.af_hidden = af_hidden
        self.dropout_hidden = dropout_hidden
        self.min_layer_width = min_layer_width
        self.max_layer_width = max_layer_width
        # member variable
        self.rng = np.random.RandomState(self.random_state)
        self.logger = get_logger(self)
        self.model = None
        self.learning_curve = [
            [],  # train_sizes_abs [0]
            [],  # train_scores    [1]
            [],  # test_scores     [2]
        ]
        self.performance_history = np.full(self.early_stopping_rounds, -np.inf)
        self.iteration_history = np.full(self.early_stopping_rounds, 0, dtype="int32")
        N = len(self.performance_history)
        self.best_estimators = np.zeros([N], dtype="object")
        if self.is_classification:
            self.score_func = accuracy_score
        else:
            self.score_func = r2_score
        self.early_stopped = False
        self.best_iteration_ = 0

    def fit(self, X, y, X_valid=None, y_valid=None, categorical_feature: Optional[List[int]] = None):
        if self.early_stopped:
            return self
        X = check_array(X)
        y = check_array(y, ensure_2d=False, dtype="float")
        if X_valid is not None:
            X_valid = check_array(X_valid)
        if y_valid is not None:
            y_valid = check_array(y_valid, ensure_2d=False, dtype="float")
        nn_param = dict(
            use_bn=self.use_bn,
            dropout_output=self.dropout_output,
            dropout_hidden=self.dropout_hidden,
            layers=(self.layer1, self.layer2),
            af_hidden=self.af_hidden,
            af_output=self.af_output,
            max_layer_width=self.max_layer_width,
            min_layer_width=self.min_layer_width
        )
        if categorical_feature is not None:
            cat_indexes = check_array(y, ensure_2d=False, dtype="int")
        else:
            cat_indexes = np.array([])
        if self.is_classification:
            n_class = None
        else:
            n_class = 1
        self.model = train_tabular_nn(
            X, y, cat_indexes, X_valid, y_valid,
            lr=self.lr,
            max_epoch=self.max_epoch,
            init_model=self.model,
            callback=self.callback, nn_params=nn_param,
            random_state=self.rng,
            n_class=n_class,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight
        )
        if self.early_stopped:
            index = int(np.lexsort((self.iteration_history, -self.performance_history))[0])
            self.best_iteration_ = int(self.iteration_history[index])
            best_estimator = self.best_estimators[index]
            self.model = best_estimator
            self.logger.info(f"{self.__class__.__name__} is early_stopped, "
                             f"best_iteration_ = {self.best_iteration_}, "
                             f"best_performace in validation_set = {self.performance_history[index]:.3f}")
            self.best_estimators = None  # do not train any more
        return self

    def predict(self, X):
        return self._predict(self.model, X)

    def _predict(self, model, X):
        X = check_array(X)
        y_pred = model(X).detach().numpy()
        if self.is_classification:
            y_pred = y_pred.argmax(axis=1)
            return y_pred
        return y_pred

    def callback(self, epoch_index, model, X, y, X_valid, y_valid) -> bool:
        if self.early_stopped:
            return self.early_stopped
        model.eval()
        should_print = self.verbose > 0 and epoch_index % self.verbose == 0
        train_score = self.score_func(y, self._predict(model, X))
        can_early_stopping = True
        if X_valid is not None:
            valid_score = self.score_func(y_valid, self._predict(model, X_valid))
        else:
            valid_score = None
            can_early_stopping = False
        score_func_name = self.score_func.__name__
        msg = f"epoch_index = {epoch_index}, " \
            f"TrainSet {score_func_name} = {train_score:.3f}"
        if valid_score is not None:
            msg += f", ValidSet {score_func_name} = {valid_score:.3f}, "
        if should_print:
            self.logger.info(msg)
        else:
            self.logger.debug(msg)
        self.learning_curve[0].append(epoch_index)
        self.learning_curve[1].append(train_score)
        self.learning_curve[2].append(valid_score)
        if can_early_stopping:
            if np.any(valid_score - self.early_stopping_tol > self.performance_history):
                index = epoch_index % self.early_stopping_rounds
                self.best_estimators[index] = deepcopy(model)
                self.performance_history[index] = valid_score
                self.iteration_history[index] = epoch_index
            else:
                self.early_stopped = True
        return self.early_stopped


class TabularNNClassifier(TabularNNEstimator, ClassifierMixin):
    is_classification = True

    def predict_proba(self, X):
        X = check_array(X)
        return self.model(X).detach().numpy()


class TabularNNRegressor(TabularNNEstimator, RegressorMixin):
    is_classification = False

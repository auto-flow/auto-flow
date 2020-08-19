#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from time import time
from typing import Dict

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from autoflow.workflow.components.base import AutoFlowComponent


class AutoFlowIterComponent(AutoFlowComponent):
    warm_start_name = "warm_start"
    iterations_name = "n_estimators"
    support_early_stopping = True

    # todo 早停后不再训练

    def __init__(self, **kwargs):
        super(AutoFlowIterComponent, self).__init__(**kwargs)
        self.init_variables()

    def init_variables(self):
        # 迭代式地训练，并引入早停机制
        if not hasattr(self, "iter_inc"):
            self.iter_inc = 10
        if not hasattr(self, "early_stopping_rounds"):
            self.early_stopping_rounds = 20
        self.performance_history = np.full(self.early_stopping_rounds, -np.inf)
        self.iteration_history = np.full(self.early_stopping_rounds, 0, dtype="int32")
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve
        self.fit_times = 0
        self.score_times = 0
        self.learning_curve = [
            [],  # train_sizes_abs [0]
            [],  # train_scores    [1]
            [],  # test_scores     [2]
            [],  # fit_times       [3]
            [],  # score_times     [4]
        ]
        N = len(self.performance_history)
        self.best_estimators = np.zeros([N], dtype="object")
        self.iter_ix = 0
        self.backup_component = None
        self.early_stopped = False

    @ignore_warnings(category=ConvergenceWarning)
    def iterative_fit(self, X, y, X_valid, y_valid, **kwargs):
        s = time()
        self.component.fit(X, y, **kwargs)
        self.fit_times += time() - s
        early_stopping_tol = getattr(self, "early_stopping_tol", 0.001)
        N = len(self.performance_history)
        if X_valid is not None and y_valid is not None:
            s = time()
            test_performance = self.component.score(X_valid, y_valid)
            train_performance = self.component.score(X, y)
            self.score_times += time() - s
            self.learning_curve[0].append(self.iteration)
            self.learning_curve[1].append(train_performance)
            self.learning_curve[2].append(test_performance)
            self.learning_curve[3].append(self.fit_times)
            self.learning_curve[4].append(self.score_times)
            if np.any(test_performance - early_stopping_tol > self.performance_history):
                index = self.iter_ix % N
                self.best_estimators[index] = deepcopy(self.component)
                self.performance_history[index] = test_performance
                self.iteration_history[index] = self.iteration
            else:
                self.early_stopped = True

    @property
    def is_fully_fitted(self):
        if (self.iteration_ >= self.max_iterations):
            self.logger.info(
                f"{self.__class__.__name__}'s next iteration "
                f"{self.iteration_ + self.iter_inc} is greater than max iterations {self.max_iterations}.")
            return True
        elif (self.early_stopped):
            self.logger.info(
                f"{self.__class__.__name__} is early stopping because "
                f"valid-set performance no longer improves. max_iter = {self.max_iterations}. "
                f"iteration = {self.iteration}")
            return True
        else:
            return False

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None, **kwargs):
        if self.backup_component is not None:
            self.component = self.backup_component
        if self.early_stopped:
            self.logger.info(f"{self.__class__.__name__} is early_stopped, keep core_fit function.")
            return self.component
        if self.best_estimators is None:
            # reload
            self.init_variables()
        while True:
            self.iterative_fit(X, y, X_valid, y_valid, **kwargs)
            if self.is_fully_fitted:
                break
            self.iteration_ = getattr(self.component, self.iterations_name)
            self.iteration_ += self.iter_inc
            # self.iteration_ = min(self.iteration_, self.max_iterations)
            setattr(self.component, self.iterations_name, int(self.iteration_))
            self.iter_ix += 1
        # todo: choose maximal performance, minimal iteration
        index = int(np.lexsort((self.iteration_history, -self.performance_history))[0])
        self.best_iteration_ = int(self.iteration_history[index])
        setattr(self, self.iterations_name, self.best_iteration_)
        best_estimator = self.best_estimators[index]
        self.backup_component = self.component
        self.component = best_estimator
        # index = np.argmax(self.performance_history)
        # self.best_estimators = None
        return self.component

    @property
    def max_iterations(self):
        return getattr(self, "max_iterations_", 1000)

    @property
    def iteration(self):
        return getattr(self, "iteration_", 1)

    def after_process_hyperparams(self, hyperparams) -> Dict:
        iter_inc = getattr(self, "iter_inc", 10)
        hyperparams = super(AutoFlowIterComponent, self).after_process_hyperparams(hyperparams)
        hyperparams[self.warm_start_name] = True
        if self.iterations_name in hyperparams:
            self.max_iterations_ = hyperparams[self.iterations_name]
        else:
            self.max_iterations_ = 1000
        hyperparams[self.iterations_name] = iter_inc
        self.iteration_ = iter_inc
        return hyperparams

    def set_max_iter(self, max_iter):
        max_iter = int(max_iter)
        if max_iter < self.hyperparams[self.iterations_name]:
            self.init_variables()
            self.component = None
        self.hyperparams[self.iterations_name] = max_iter
        self.set_inside_dict({self.iterations_name: max_iter})
        self.max_iterations_ = max_iter

    def finish_evaluation(self):
        self.best_estimators = None


class LgbmIterativeMixIn():
    def set_max_iter(self, max_iter):
        max_iter = int(max_iter)
        if max_iter < self.hyperparams["n_estimators"]:
            self.component = None
        self.hyperparams["n_estimators"] = max_iter
        if self.component is not None:
            self.component.n_estimators = max_iter

    def finish_evaluation(self):
        pass


class TabularNNIterativeMixIn():
    def set_max_iter(self, max_iter):
        max_iter = int(max_iter)
        if max_iter < self.hyperparams["max_epoch"]:
            self.component = None
        self.hyperparams["max_epoch"] = max_iter
        if self.component is not None:
            self.component.max_epoch = max_iter

    def finish_evaluation(self):
        if self.component is not None:
            self.component.best_estimators = None
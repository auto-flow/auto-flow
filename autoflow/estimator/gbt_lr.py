#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from time import time

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._testing import ignore_warnings

from autoflow.estimator.wrap_lightgbm import LGBMClassifier


class GBTLRClassifier(LGBMClassifier):
    def __init__(
            self,
            n_estimators=256,
            objective=None,
            boosting_type="gbdt",
            learning_rate=0.01,
            max_depth=31,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            random_state=0,
            lambda_l1=0.1,
            lambda_l2=0.2,
            subsample_for_bin=40000,
            min_child_weight=0.01,
            early_stopping_rounds=250,
            verbose=-1,
            n_jobs=1,
            # lr params
            penalty="l2",
            # solver="saga",
            # l1_ratio=0.5,
            C=0.01,
            max_iter=600,
            iter_step=20,
            lr_es_round=4
    ):
        super(GBTLRClassifier, self).__init__(
            n_estimators,
            objective,
            boosting_type,
            learning_rate,
            max_depth,
            num_leaves,
            feature_fraction,
            bagging_fraction,
            bagging_freq,
            random_state,
            lambda_l1,
            lambda_l2,
            subsample_for_bin,
            min_child_weight,
            early_stopping_rounds,
            verbose,
            n_jobs,
        )
        self.lr_es_round = lr_es_round
        self.iter_step = iter_step
        self.max_iter = max_iter
        self.C = C
        # self.l1_ratio = l1_ratio
        # self.solver = solver
        self.penalty = penalty

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y, X_valid=None, y_valid=None, categorical_feature="auto",
            sample_weight=None, **kwargs):
        gbt_start = time()
        super(GBTLRClassifier, self).fit(
            X=X,
            y=y,
            X_valid=X_valid,
            y_valid=y_valid,
            categorical_feature=categorical_feature,
            sample_weight=sample_weight
        )
        gbt_end = time()
        leaf = self.model.predict(X, pred_leaf=True)
        ohe_start = time()
        self.ohe = OneHotEncoder(dtype="int32", handle_unknown="ignore").fit(leaf)
        feature = self._transform_leaf(leaf)
        ohe_end = time()
        # 对LR做递增的，早停训练
        self.performance_history = np.full(self.lr_es_round, -np.inf)
        self.iteration_history = np.full(self.lr_es_round, 0, dtype="int32")
        self.best_estimators = np.zeros([self.lr_es_round], dtype="object")
        if X_valid is not None and y_valid is not None:
            can_es_lr = True
            X_valid_transform = self.transform(X_valid)
        else:
            can_es_lr = False
            X_valid_transform = None
        lr_start = time()
        lr = LogisticRegression(
            penalty=self.penalty,
            # solver=self.solver,
            C=self.C,
            # l1_ratio=self.l1_ratio,
            max_iter=self.iter_step,
            warm_start=True
        )
        for i, max_iter in enumerate(range(self.iter_step, self.max_iter, self.iter_step)):
            lr.max_iter = max_iter
            lr.fit(feature, y, sample_weight=sample_weight)
            if can_es_lr:
                score = lr.score(X_valid_transform, y_valid)
                if np.any(score > self.performance_history):
                    index = i % self.lr_es_round
                    self.best_estimators[index] = deepcopy(lr)
                    self.performance_history[index] = score
                    self.iteration_history[index] = max_iter
                else:
                    break
        if can_es_lr:
            index = int(np.lexsort((self.iteration_history, -self.performance_history))[0])
            self.lr_best_iteration = int(self.iteration_history[index])
            self.lr = self.best_estimators[index]
            self.logger.info(
                f"{self.__class__.__name__}'s {self.lr.__class__.__name__} early_stopped, best_iteration_ = {self.lr_best_iteration}")
        else:
            self.lr_best_iteration = self.max_iter
            self.lr = lr
        lr_end = time()
        self.gbt_cost_time = gbt_end - gbt_start
        self.ohe_cost_time = ohe_end - ohe_start
        self.lr_cost_time = lr_end - lr_start
        self.logger.info(f"fit GBT cost {self.gbt_cost_time:.3f}s , "
                         f"fit OHE cost {self.ohe_cost_time:.3f}s , "
                         f"fit LR cost {self.lr_cost_time:.3f}s . ")
        return self

    def _transform_leaf(self, leaf):
        return self.ohe.transform(leaf)

    def transform(self, X):
        leaf = self.model.predict(X, pred_leaf=True)
        feature = self._transform_leaf(leaf)
        return feature

    def predict_proba(self, X):
        return self.lr.predict_proba(self.transform(X))

    def predict(self, X):
        return self.lr.predict(self.transform(X))

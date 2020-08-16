#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import warnings
from collections import Counter
from time import time

import numpy as np
from frozendict import frozendict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import check_X_y, check_array
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.multiclass import type_of_target

from autoflow.utils.data import check_n_jobs
from autoflow.utils.logging_ import get_logger

warnings.simplefilter("ignore")


class AdaptiveScaler(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            budget_per_trial=1,
            budget=10,
            # n_jobs=-1,
            verbose=0,
            random_state=42,
            cv=3,
            lr_iter_step=10,
            lr_max_iter=100,
            lr_es_round=4,

            problem_type=None
    ):
        self.lr_es_round = lr_es_round
        self.lr_max_iter = lr_max_iter
        self.lr_iter_step = lr_iter_step
        self.problem_type = problem_type
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        # self.n_jobs = check_n_jobs(n_jobs)
        self.budget = budget
        self.budget_per_trial = budget_per_trial
        self.logging_level = 20 if verbose > 0 else 20
        self.logger = get_logger(self)

    def method2scaler_cls(self, method):
        if method == "standard":
            return StandardScaler
        elif method == "robust":
            return RobustScaler
        else:
            raise ValueError(f"Unknown method '{method}'")

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_test=None):
        if self.problem_type is None:
            if type_of_target(y_train) == "continuous":
                self.problem_type = "regression"
            else:
                self.problem_type = "classification"
        # check input variables
        X_train, y_train = check_X_y(X_train, y_train, y_numeric=self.problem_type == "regression", dtype=None)
        X_train_pool = [X_train]
        y_train_pool = [y_train]
        X_pool = [X_train]
        if X_valid is not None:
            X_valid, y_valid = check_X_y(X_valid, y_valid, y_numeric=self.problem_type == "regression", dtype=None)
            self.logger.log(self.logging_level, f"X_valid and y_valid exist. shape = {X_train.shape} .")
            X_train_pool.append(X_valid)
            y_train_pool.append(y_train)
        if X_test is not None:
            X_test = check_array(X_test)
            self.logger.log(self.logging_level, f"X_test exist. shape = {X_test.shape} .")
            X_pool.append(X_test)
        X_train = np.vstack(X_train_pool)
        X = np.vstack(X_pool)
        y_train = np.hstack(y_train_pool)
        if self.problem_type == "classification":
            self.kfold = StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        else:
            self.kfold = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        candidates = [
            ["standard", {}],
        ]
        self.scaler2info = {}
        q_mins = (0.1, 0.25)
        q_maxs = (0.75, 0.9)
        neighbors = {
            0.1: [0.05, 0.15],
            0.25: [0.2, 0.3],
            0.9: [0.85, 0.95],
            0.75: [0.7, 0.8],
        }
        start_time = time()
        for q_min in q_mins:
            for q_max in q_maxs:
                candidates.append(["robust", {"quantile_range": (q_min, q_max)}])
        results = []
        for (method, params) in candidates:
            results.append(
                self.evaluate(X_train, y_train, method, params)
            )
            if time() - start_time > self.budget:
                break
        best_score = -np.inf
        best_scaler = None
        self.logger.log(self.logging_level, '-' * 50)
        cv_cnts = []
        for (method, params), result in zip(candidates, results):
            score = result["score"]
            self.scaler2info[(method, frozendict(params))] = result
            mark = ""
            if score > best_score:
                if best_scaler is not None and best_scaler[0] == "standard" and score - best_score < 0.002:
                    self.logger.log(self.logging_level,
                                    f"best_scaler = standard, score = {score:.3f}, best_score = {best_score:.3f}, cannot challenge incumbent")
                else:
                    best_score = score
                    best_scaler = (method, params)
                    mark = "* "
            msg0 = f'1 {mark}| score = {score:.3f} , time = {result["eval_cost_time"]:.2f} , method = {method}'
            if method == "robust":
                q_min, q_max = params[list(params.keys())[0]]
                msg0 += f" , q_min = {q_min:.2f}, q_max = {q_max:.2f}"
            msg0 += f" , max_iters = {result['max_iters']}"
            cv_cnts.append(result["cv_cnt"])
            self.logger.log(self.logging_level, msg0)

        # find best scaler again
        if best_scaler[0] == "robust" and time() - start_time < self.budget // 2:
            self.logger.log(self.logging_level, '-' * 50)
            q_mins = neighbors[best_scaler[1]['quantile_range'][0]]
            q_maxs = neighbors[best_scaler[1]['quantile_range'][1]]
            candidates = []
            for q_min in q_mins:
                for q_max in q_maxs:
                    candidates.append(["robust", {"quantile_range": (q_min, q_max)}])
            results = []
            for (method, params) in candidates:
                results.append(
                    self.evaluate(X_train, y_train, method, params)
                )
                if time() - start_time > self.budget:
                    break
            for (method, params), result in zip(candidates, results):
                score = result["score"]
                self.scaler2info[(method, frozendict(params))] = result
                mark = ""
                if score > best_score:
                    best_score = score
                    best_scaler = (method, params)
                    mark = "* "
                msg0 = f'2 {mark}| score = {score:.3f} , time = {result["eval_cost_time"]:.2f} , method = {method}'
                if method == "robust":
                    q_min, q_max = params[list(params.keys())[0]]
                    msg0 += f" , q_min = {q_min:.2f}, q_max = {q_max:.2f}"
                msg0 += f" , max_iters = {result['max_iters']}"
                cv_cnts.append(result["cv_cnt"])
                self.logger.log(self.logging_level, msg0)

        self.logger.log(self.logging_level, '-' * 50)
        cv_cnt_counter = Counter(cv_cnts)
        msg1 = "'cv_cnt' statistics: "
        for cv_cnt, times in cv_cnt_counter.items():
            msg1 += f"{cv_cnt}-cv [{times} times]; "
        self.logger.log(self.logging_level, msg1)
        self.best_scaler = best_scaler
        self.best_score = best_score
        self.scaler = self.method2scaler_cls(best_scaler[0])(**best_scaler[1])
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X = check_array(X)
        return self.scaler.transform(X)

    def get_model(self):
        if self.problem_type == "regression":
            return ElasticNet(
                random_state=self.random_state,
                normalize=True,
                warm_start=True
            )
        elif self.problem_type == "classification":
            return LogisticRegression(
                random_state=self.random_state,
                n_jobs=1,
                warm_start=True,
                penalty="l2"
            )
        else:
            raise ValueError(f"Unknown problem-type {self.problem_type}")

    @ignore_warnings(category=ConvergenceWarning)
    def evaluate(self, X, y, method, params):
        start_time = time()
        cv_cnt = 0
        scores = []
        max_iters = []

        for train_ix, valid_ix in self.kfold.split(X, y):
            X_train = X[train_ix, :]
            y_train = y[train_ix]
            X_valid = X[valid_ix, :]
            y_valid = y[valid_ix]
            pipeline = Pipeline([
                ("scaler", self.method2scaler_cls(method)(**params)),
                ("linear_model", self.get_model())  # todo: regression modeling
            ])
            should_break = False
            performance_history = np.full(self.lr_es_round, -np.inf)
            for i, max_iter in enumerate(range(self.lr_iter_step, self.lr_max_iter, self.lr_iter_step)):
                pipeline[-1].max_iter = max_iter
                pipeline.fit(X_train, y_train)
                score = pipeline.score(X_valid, y_valid)
                if np.any(score > performance_history):
                    index = i % self.lr_es_round
                    performance_history[index] = score
                else:
                    break
                if time() - start_time > self.budget_per_trial:
                    should_break = True
                    break
            scores.append(performance_history.max())
            max_iters.append(pipeline[-1].max_iter)
            cv_cnt += 1
            if time() - start_time > self.budget_per_trial:
                should_break = True
            if should_break:
                break
        eval_cost_time = time() - start_time
        return {
            "eval_cost_time": eval_cost_time,
            "score": float(np.mean(scores)),
            "scores": scores,
            "max_iters": max_iters,
            "cv_cnt": cv_cnt
        }

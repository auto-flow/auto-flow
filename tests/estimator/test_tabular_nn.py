#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
from sklearn.datasets import load_digits, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from autoflow.estimator.tabular_nn_est import TabularNNClassifier, TabularNNRegressor
from autoflow.tests.base import EstimatorTestCase
from autoflow.utils.logging_ import setup_logger

setup_logger()


class TestTabularNNEstimator(EstimatorTestCase):
    current_file = __file__

    def test_adult_dataset(self):
        tabular = TabularNNClassifier(
            verbose=1, max_epoch=32, early_stopping_rounds=8, n_jobs=-1,  # , class_weight="balanced"
        )
        tabular.fit(self.X_train, self.y_train, self.X_test, self.y_test, categorical_feature=self.cat_indexes.tolist())
        print(tabular.score(self.X_test, self.y_test))

    def test_multiclass(self):
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        tabular = TabularNNClassifier(
            verbose=1, max_epoch=64, early_stopping_rounds=16, n_jobs=-1, lr=1e-2
        )
        tabular.fit(X_train, y_train, X_test, y_test)
        print(tabular.score(X_test, y_test))
        y_score = tabular.predict_proba(X_test)
        assert y_score.shape[1] == 10
        assert np.all(np.abs(y_score.sum(axis=1) - 1) < 1e3)
        if tabular.early_stopped:
            assert tabular.best_estimators is None

    def test_boston(self):
        X, y = load_boston(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        # y = StandardScaler().fit_transform(y[:, None]).flatten()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        tabular = TabularNNRegressor(
            verbose=1, max_epoch=64, early_stopping_rounds=16, n_jobs=-1,lr=1e-2
        )
        tabular.fit(X_train, y_train, X_test, y_test)
        print(tabular.score(X_test, y_test))

    def test_warm_start(self):
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        tabular = TabularNNClassifier(
            verbose=1, max_epoch=64, early_stopping_rounds=16, n_jobs=-1
        )
        for max_epoch in [4, 16, 64]:
            tabular.max_epoch = max_epoch
            tabular.fit(X_train, y_train, X_test, y_test)
            score = tabular.score(X_test, y_test)
            tabular.logger.info(f"max_epoch = {max_epoch}, score = {score:.3f}")

    def test_no_valid_set(self):
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        tabular = TabularNNClassifier(
            verbose=1, max_epoch=64, early_stopping_rounds=16, n_jobs=-1
        )
        tabular.fit(X_train, y_train)
        score = tabular.score(X_test, y_test)
        print(score)

    def test_cat_only(self):
        X_train=self.X_train.values[:,self.cat_indexes]
        X_test=self.X_test.values[:,self.cat_indexes]
        cat_indexes=np.arange(X_train.shape[1],dtype="int")
        tabular = TabularNNClassifier(
            verbose=1, max_epoch=16, early_stopping_rounds=8, n_jobs=-1
        )
        tabular.fit(X_train, self.y_train, categorical_feature=cat_indexes.tolist())
        score = tabular.score(X_test, self.y_test)

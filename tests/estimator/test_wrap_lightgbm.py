#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

import numpy as np
from sklearn.datasets import load_digits, load_boston
from sklearn.model_selection import train_test_split

from autoflow.estimator.wrap_lightgbm import LGBMRegressor, LGBMClassifier
from autoflow.tests.base import EstimatorTestCase


def calc_balanced_sample_weight(y_train: np.ndarray):
    unique, counts = np.unique(y_train, return_counts=True)
    # This will result in an average weight of 1!
    cw = 1 / (counts / np.sum(counts)) / len(unique)

    sample_weights = np.ones(y_train.shape)

    for i, ue in enumerate(unique):
        mask = y_train == ue
        sample_weights[mask] *= cw[i]
    return sample_weights


class TestWrapLightGBM(EstimatorTestCase):
    current_file = __file__

    def test_multiclass(self):
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        lgbm = LGBMClassifier(n_estimators=5000, verbose=100)
        lgbm.fit(X_train, y_train, X_test, y_test)
        print(lgbm.score(X_test, y_test))
        y_score = lgbm.predict_proba(X_test)
        assert y_score.shape[1] == 10
        assert np.all(np.abs(y_score.sum(axis=1) - 1) < 1e5)

    def test_regression(self):
        X, y = load_boston(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        lgbm = LGBMRegressor(n_estimators=5000, verbose=100)
        lgbm.fit(X_train, y_train, X_test, y_test)
        print(lgbm.score(X_test, y_test))

    def test_no_validate_set(self):
        lgbm = LGBMClassifier(n_estimators=100, verbose=10)
        lgbm.fit(self.X_train, self.y_train)
        print(lgbm.score(self.X_test, self.y_test))

    def test_warm_start(self):
        lgbm = LGBMClassifier(verbose=16)
        # 0.8764 1618
        # 0.8749 1557
        for n_estimators in [128, 512, 2048, 4096]:
            lgbm.n_estimators = n_estimators
            lgbm.fit(self.X_train, self.y_train, self.X_test, self.y_test)
            acc = lgbm.score(self.X_test, self.y_test)
            print(f"n_estimator = {n_estimators}, accuracy = {acc:.4f}")

    def test_use_categorical_feature(self):
        # 测试category
        lgbm = LGBMClassifier(n_estimators=2000, verbose=100)
        lgbm.fit(self.X_train, self.y_train, self.X_test, self.y_test, categorical_feature=self.cat_indexes.tolist())
        print(lgbm.score(self.X_test, self.y_test))

    def test_sample_weight(self):
        lgbm = LGBMClassifier(n_estimators=2000, verbose=100)
        sample_weight = calc_balanced_sample_weight(self.y_train)
        lgbm.fit(self.X_train, self.y_train, self.X_test, self.y_test, sample_weight=sample_weight)
        print(lgbm.score(self.X_test, self.y_test))

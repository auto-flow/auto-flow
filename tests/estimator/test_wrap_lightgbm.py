#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from pathlib import Path
from unittest import TestCase

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_boston
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder

from autoflow.datasets import load_task
from autoflow.estimator.wrap_lightgbm import LGBMRegressor, LGBMClassifier


def calc_balanced_sample_weight(y_train: np.ndarray):
    unique, counts = np.unique(y_train, return_counts=True)
    # This will result in an average weight of 1!
    cw = 1 / (counts / np.sum(counts)) / len(unique)

    sample_weights = np.ones(y_train.shape)

    for i, ue in enumerate(unique):
        mask = y_train == ue
        sample_weights[mask] *= cw[i]
    return sample_weights


class TestWrapLightGBM(TestCase):
    def setUp(self) -> None:
        cur_dir = Path(__file__).parent
        if (cur_dir / "126025.bz2").exists():
            X_train, y_train, X_test, y_test, cat = joblib.load(cur_dir / "126025.bz2")
        else:
            X_train, y_train, X_test, y_test, cat = load_task(126025)
            joblib.dump(
                [X_train, y_train, X_test, y_test, cat],
                cur_dir / "126025.bz2"
            )
        nan_cnt = np.count_nonzero(pd.isna(pd.concat([X_train, X_test])), axis=0)
        cat = np.array(cat)
        cat_na_mask = (nan_cnt > 0) & cat
        num_na_mask = (nan_cnt > 0) & (~cat)
        cat_imputer = SimpleImputer(strategy="constant", fill_value="NA").fit(X_train.loc[:, cat_na_mask])
        # num_imputer = SimpleImputer(strategy="median").fit(X_train.loc[:, num_na_mask])
        X_train.loc[:, cat_na_mask] = cat_imputer.transform(X_train.loc[:, cat_na_mask])
        X_test.loc[:, cat_na_mask] = cat_imputer.transform(X_test.loc[:, cat_na_mask])
        # X_train.loc[:, num_na_mask] = num_imputer.transform(X_train.loc[:, num_na_mask])
        # X_test.loc[:, num_na_mask] = num_imputer.transform(X_test.loc[:, num_na_mask])
        ordinal_encoder = OrdinalEncoder(dtype="int").fit(X_train.loc[:, cat])
        transformer = StandardScaler().fit(X_train.loc[:, ~cat])
        X_train.loc[:, cat] = ordinal_encoder.transform(X_train.loc[:, cat])
        X_train.loc[:, ~cat] = transformer.transform(X_train.loc[:, ~cat])
        X_test.loc[:, cat] = ordinal_encoder.transform(X_test.loc[:, cat])
        X_test.loc[:, ~cat] = transformer.transform(X_test.loc[:, ~cat])
        self.cat_indexes = np.arange(len(cat))[cat]
        label_encoder = LabelEncoder().fit(y_train)
        self.y_train = label_encoder.transform(y_train)
        self.y_test = label_encoder.transform(y_test)
        self.X_train = X_train
        self.X_test = X_test

    def test_multiclass(self):
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        lgbm = LGBMClassifier(n_estimators=5000, verbose=100)
        lgbm.fit(X_train, y_train, X_test, y_test)
        print(lgbm.score(X_test, y_test))

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
        for n_estimator in [128, 512, 2048, 4096]:
            lgbm.n_estimators = n_estimator
            lgbm.fit(self.X_train, self.y_train, self.X_test, self.y_test)
            acc = lgbm.score(self.X_test, self.y_test)
            print(f"n_estimator = {n_estimator}, accuracy = {acc:.4f}")

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

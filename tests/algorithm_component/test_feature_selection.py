#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from time import time

import numpy as np

from autoflow.data_container import DataFrameContainer, NdArrayContainer
from autoflow.datasets import load
from autoflow.tests.base import LocalResourceTestCase
from autoflow.workflow.components.preprocessing.select.from_model_clf import SelectFromModelClf
from autoflow.workflow.components.preprocessing.select.from_model_reg import SelectFromModelReg
from autoflow.workflow.components.preprocessing.select.rfe_clf import RFE_Clf
from autoflow.workflow.components.preprocessing.select.rfe_reg import RFE_Reg
from autoflow.workflow.components.preprocessing.select.univar_clf import SelectPercentileClassification
from autoflow.workflow.components.preprocessing.select.univar_reg import SelectPercentileRegression


def very_close(i, j, delta=2):
    if abs(i - j) < delta:
        return True
    return False


class TestFeatureSelection(LocalResourceTestCase):
    def setUp(self) -> None:
        super(TestFeatureSelection, self).setUp()
        self.L = 1024
        df = load("qsar")
        y = df.pop("target")
        X = df
        X[X == 0] = -1
        X.index = reversed(X.index)
        self.index = deepcopy(X.index)
        X = DataFrameContainer("TrainSet", dataset_instance=X, resource_manager=self.mock_resource_manager)
        X.set_feature_groups(["num"] * X.shape[1])
        self.X = X
        self.y = NdArrayContainer("TrainSet", dataset_instance=y, resource_manager=self.mock_resource_manager)
        y_reg = y + np.random.rand(*y.shape)
        self.y_reg = NdArrayContainer("TrainSet", dataset_instance=y_reg, resource_manager=self.mock_resource_manager)

    def do_assert(self, trans_X, percent):
        assert np.all(trans_X.index == self.index)
        assert np.all(trans_X.feature_groups == "selected")
        assert very_close(int(self.L * (percent / 100)), trans_X.shape[1])
        assert len(set(trans_X.columns) & set(self.X.columns)) \
               == len(trans_X.columns)
        assert np.all(trans_X.data == self.X.data[trans_X.columns])

    def test_univar_clf(self):
        for p in [50, 100]:
            for f in ["chi2", "f_classif", "mutual_info_classif"]:
                kwargs = {
                    "score_func": f,
                    "_select_percent": p
                }
                selector = SelectPercentileClassification(**kwargs)
                selector.in_feature_groups = "num"
                selector.out_feature_groups = "selected"
                start = time()
                print(f"testing percent = {p}, score_func = {f}")
                trans_X = selector.fit_transform(self.X, self.y)["X_train"]
                print(f"time consuming {time() - start:.4f}s")
                self.do_assert(trans_X, p)

    def test_univar_reg(self):
        for p in [50, 100]:
            for f in ["f_regression", "mutual_info_regression"]:
                kwargs = {
                    "score_func": f,
                    "_select_percent": p
                }
                selector = SelectPercentileRegression(**kwargs)
                selector.in_feature_groups = "num"
                selector.out_feature_groups = "selected"
                start = time()
                print(f"testing percent = {p}, score_func = {f}")
                trans_X = selector.fit_transform(self.X, self.y_reg)["X_train"]
                print(f"time consuming {time() - start:.4f}s")
                self.do_assert(trans_X, p)

    def test_from_model_clf(self):
        for p in [50, 100]:
            for m in [
                "sklearn.ensemble.ExtraTreesClassifier",
                "sklearn.svm.LinearSVC",
                "sklearn.linear_model.LogisticRegression",
            ]:
                kwargs = {
                    "estimator": m,
                    "_select_percent": p,
                    "penalty": "l1",
                    "dual": False,
                    "multi_class": "ovr",
                    "C": 1,
                    "n_estimators": 10,
                    "max_depth": 7,
                    "min_samples_split": 10,
                    "min_samples_leaf": 10,
                    "random_state": 42,
                    "n_jobs": -1,
                    "solver": "saga",
                }
                selector = SelectFromModelClf(**kwargs)
                selector.in_feature_groups = "num"
                selector.out_feature_groups = "selected"
                start = time()
                print(f"testing percent = {p}, base_model = {m}")
                trans_X = selector.fit_transform(self.X, self.y)["X_train"]
                print(f"time consuming {time() - start:.4f}s")
                self.do_assert(trans_X, p)

    def test_from_model_reg(self):
        for p in [50, 100]:
            for m in [
                "sklearn.ensemble.ExtraTreesRegressor",
                "sklearn.svm.LinearSVR",
                "sklearn.linear_model.Ridge",
                "sklearn.linear_model.Lasso",
            ]:
                kwargs = {
                    "estimator": m,
                    "_select_percent": p,
                    "loss": "epsilon_insensitive",
                    "dual": True,
                    "C": 1,
                    "n_estimators": 10,
                    "max_depth": 7,
                    "min_samples_split": 10,
                    "min_samples_leaf": 10,
                    "random_state": 42,
                    "n_jobs": -1,
                }
                selector = SelectFromModelReg(**kwargs)
                selector.in_feature_groups = "num"
                selector.out_feature_groups = "selected"
                start = time()
                print(f"testing percent = {p}, base_model = {m}")
                trans_X = selector.fit_transform(self.X, self.y_reg)["X_train"]
                print(f"time consuming {time() - start:.4f}s")
                self.do_assert(trans_X, p)

    def test_ref_clf(self):
        for p in [50, 100]:
            for m in [
                "sklearn.ensemble.ExtraTreesClassifier",
                "sklearn.svm.LinearSVC",
                "sklearn.linear_model.LogisticRegression",
            ]:
                kwargs = {
                    "estimator": m,
                    "_select_percent": p,
                    "_step__sp1_dev": 20,
                    "penalty": "l1",
                    "dual": False,
                    "multi_class": "ovr",
                    "C": 1,
                    "n_estimators": 10,
                    "max_depth": 7,
                    "min_samples_split": 10,
                    "min_samples_leaf": 10,
                    "random_state": 42,
                    "n_jobs": -1,
                    "solver": "saga",
                }
                selector = RFE_Clf(**kwargs)
                selector.in_feature_groups = "num"
                selector.out_feature_groups = "selected"
                start = time()
                print(f"testing percent = {p}, base_model = {m}")
                trans_X = selector.fit_transform(self.X, self.y)["X_train"]
                print(f"time consuming {time() - start:.4f}s")
                self.do_assert(trans_X, p)

    def test_ref_reg(self):
        for p in [50, 100]:
            for m in [
                "sklearn.ensemble.ExtraTreesRegressor",
                "sklearn.svm.LinearSVR",
                "sklearn.linear_model.Ridge",
                "sklearn.linear_model.Lasso",
            ]:
                kwargs = {
                    "estimator": m,
                    "_select_percent": p,
                    "_step__sp1_dev": 20,
                    "loss": "epsilon_insensitive",
                    "dual": True,
                    "C": 1,
                    "n_estimators": 10,
                    "max_depth": 7,
                    "min_samples_split": 10,
                    "min_samples_leaf": 10,
                    "random_state": 42,
                    "n_jobs": -1,
                }
                selector = RFE_Reg(**kwargs)
                selector.in_feature_groups = "num"
                selector.out_feature_groups = "selected"
                start = time()
                print(f"testing percent = {p}, base_model = {m}")
                trans_X = selector.fit_transform(self.X, self.y_reg)["X_train"]
                print(f"time consuming {time() - start:.4f}s")
                self.do_assert(trans_X, p)

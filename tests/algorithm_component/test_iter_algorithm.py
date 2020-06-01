#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from time import time

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from autoflow.data_container import DataFrameContainer
from autoflow.data_container import NdArrayContainer
from autoflow.hdl.utils import get_default_hdl_bank
from autoflow.test.base import LocalResourceTestCase
from autoflow.workflow.components.classification.extra_trees import ExtraTreesClassifier
from autoflow.workflow.components.classification.gradient_boosting import GradientBoostingClassifier
from autoflow.workflow.components.classification.logistic_regression import LogisticRegression
from autoflow.workflow.components.classification.random_forest import RandomForestClassifier
from autoflow.workflow.components.classification.sgd import SGDClassifier
from autoflow.workflow.components.regression.elasticnet import ElasticNet
from autoflow.workflow.components.regression.extra_trees import ExtraTreesRegressor
from autoflow.workflow.components.regression.gradient_boosting import GradientBoostingRegressor
from autoflow.workflow.components.regression.random_forest import RandomForestRegressor
from autoflow.workflow.components.regression.sgd import SGDRegressor


def get_hp_of_cls(cls, hdl_bank, key1):
    module = cls.__module__
    key2 = module.split(".")[-1]
    hp = deepcopy(hdl_bank[key1][key2])
    for k, v in hp.items():
        if isinstance(v, dict) and "_default" in v:
            hp[k] = v["_default"]
    return hp


class TestIterAlforithm(LocalResourceTestCase):

    def test_classifier(self):
        X, y = datasets.load_digits(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=0)
        X_train = DataFrameContainer("TrainSet", dataset_instance=X_train, resource_manager=self.mock_resource_manager)
        X_test = DataFrameContainer("TestSet", dataset_instance=X_test, resource_manager=self.mock_resource_manager)
        y_train = NdArrayContainer("TrainLabel", dataset_instance=y_train, resource_manager=self.mock_resource_manager)
        y_test = NdArrayContainer("TestLabel", dataset_instance=y_test, resource_manager=self.mock_resource_manager)

        hdl_bank = get_default_hdl_bank()
        est_cls_list = [
            LogisticRegression,
            GradientBoostingClassifier,
            RandomForestClassifier,
            ExtraTreesClassifier,
            SGDClassifier,
        ]
        for cls in est_cls_list:
            print("=========================")
            print(cls.__name__)
            print("=========================")
            est = cls(
                **get_hp_of_cls(cls, hdl_bank, "classification")
            )
            start = time()
            est.fit(X_train, y_train, X_test, y_test)
            score = est.component.score(X_test.data, y_test.data)
            end = time()
            print("score:", score)
            print("time:", end - start)
            self.assertTrue(score == np.max(est.performance_history))
            print("max_iterations:", est.max_iterations)
            print("early_stopping_rounds:", est.early_stopping_rounds)
            print("early_stopping_tol:", est.early_stopping_tol)
            print("iter_inc:", est.iter_inc)
            print("iteration:", est.iteration)
            print("iter_ix:", est.iter_ix)
            print("min_performance:", np.min(est.performance_history))
            print("max_performance:", np.max(est.performance_history))
            print('\n' * 2)

    def test_regressor(self):
        X, y = datasets.load_boston(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=0)
        X_train = DataFrameContainer("TrainSet", dataset_instance=X_train, resource_manager=self.mock_resource_manager)
        X_test = DataFrameContainer("TestSet", dataset_instance=X_test, resource_manager=self.mock_resource_manager)
        y_train = NdArrayContainer("TrainLabel", dataset_instance=y_train, resource_manager=self.mock_resource_manager)
        y_test = NdArrayContainer("TestLabel", dataset_instance=y_test, resource_manager=self.mock_resource_manager)

        hdl_bank = get_default_hdl_bank()
        est_cls_list = [
            ElasticNet,
            GradientBoostingRegressor,
            RandomForestRegressor,
            ExtraTreesRegressor,
            SGDRegressor
        ]
        for cls in est_cls_list:
            print("=========================")
            print(cls.__name__)
            print("=========================")
            est = cls(
                **get_hp_of_cls(cls, hdl_bank, "regression")
            )
            start = time()
            est.fit(X_train, y_train, X_test, y_test)
            score = est.component.score(X_test.data, y_test.data)
            end = time()
            print("score:", score)
            print("time:", end - start)
            if cls.__name__ != "SGDRegressor":
                self.assertTrue(score == np.max(est.performance_history))
            print("max_iterations:", est.max_iterations)
            print("early_stopping_rounds:", est.early_stopping_rounds)
            print("early_stopping_tol:", est.early_stopping_tol)
            print("iter_inc:", est.iter_inc)
            print("iteration:", est.iteration)
            print("iter_ix:", est.iter_ix)
            print("min_performance:", np.min(est.performance_history))
            print("max_performance:", np.max(est.performance_history))
            print('\n' * 2)

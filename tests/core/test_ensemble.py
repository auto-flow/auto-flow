#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold  #, LeaveOneOut

from autoflow import AutoFlowRegressor
from autoflow.core.classifier import AutoFlowClassifier
from autoflow.tests.base import LocalResourceTestCase


class TestEnsemble(LocalResourceTestCase):
    def test_ensemble_classifiers(self):
        X, y = load_iris(return_X_y=True)
        y = y.astype("str")
        y[y == '0'] = "apple"
        y[y == '1'] = "pear"
        y[y == '2'] = "banana"
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        pipe = AutoFlowClassifier(
            DAG_workflow={
                "num->target": ["liblinear_svc", "libsvm_svc", "logistic_regression"]
            },
            initial_runs=2,
            run_limit=2,
            n_jobs=2,
            debug=True,
        )
        pipe.fit(X_train, y_train, splitter=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42))
        score = pipe.score(X_test, y_test)
        print(score)
        assert pipe.estimator.stacked_y_true.shape == (46,)
        assert np.all(pipe.estimator.prediction_list[0].sum(axis=1) - 1 < 0.001)
        assert pipe.estimator.prediction_list[0].shape == (46, 3)
        assert score > 0.9
        for splitter in [
            # LeaveOneOut(),
            ShuffleSplit(n_splits=20, test_size=0.3, random_state=42),
            KFold()
        ]:
            pipe.fit(X_train, y_train, splitter=splitter)
            score = pipe.score(X_test, y_test)
            assert score > 0.9
            print("splitter:", splitter)
            print("test accuracy:", score)

    def test_ensemble_regressors(self):
        X, y = load_boston(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        pipe = AutoFlowRegressor(
            DAG_workflow={
                "num->scaled": ["scale.standardize"],
                "scaled->target": ["elasticnet"]
            },
            initial_runs=2,
            run_limit=2,
            n_jobs=2,
            debug=True,
        )
        pipe.fit(X_train, y_train, splitter=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42))
        score = pipe.score(X_test, y_test)
        print(score)
        assert score > 0.5
        for splitter in [
            # LeaveOneOut(),
            ShuffleSplit(n_splits=20, test_size=0.3, random_state=42),
            KFold()
        ]:
            pipe.fit(X_train, y_train, splitter=splitter)
            score = pipe.score(X_test, y_test)
            assert score > 0.5
            print("splitter:", splitter)
            print("test r2:", score)

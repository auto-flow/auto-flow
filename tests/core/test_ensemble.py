#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy

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
                "num->target": ["linearsvc", "svc", "logistic_regression"]
            },
            initial_runs=2,
            run_limit=2,
            n_jobs=2,
            resource_manager=self.mock_resource_manager,
            debug=True,
        )
        pipe.fit(X_train, y_train, splitter=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42))
        score = pipe.score(X_test, y_test)
        print(score)
        assert pipe.estimator.stacked_y_true.shape == (46,)
        assert np.all(pipe.estimator.prediction_list[0].sum(axis=1) - 1 < 0.001)
        assert pipe.estimator.prediction_list[0].shape == (46, 3)
        assert score > 0.8
        for splitter in [
            # LeaveOneOut(),
            ShuffleSplit(n_splits=20, test_size=0.3, random_state=42),
            KFold()
        ]:
            pipe.fit(X_train, y_train, splitter=splitter)
            score = pipe.score(X_test, y_test)
            assert score > 0.8
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
            resource_manager=self.mock_resource_manager,
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

class TestEnsembleAlone(LocalResourceTestCase):
    def test(self):
        X, y = load_iris(return_X_y=True)
        y = y.astype("str")
        y[y == '0'] = "apple"
        y[y == '1'] = "pear"
        y[y == '2'] = "banana"
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        pipe = AutoFlowClassifier(
            DAG_workflow={
                "num->target": ["linearsvc", "svc", "logistic_regression"]
            },
            initial_runs=6,
            run_limit=6,
            n_jobs=2,
            debug=True,
            resource_manager=self.mock_resource_manager
        )
        pipe.fit(
            X_train, y_train,
            splitter=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            fit_ensemble_params=False
        )
        assert pipe.experiment_id == 1
        data_manager = pipe.data_manager.copy(keep_data=False)
        assert data_manager.X_train is None
        assert pipe.data_manager.X_train is not None
        #######################################################
        ensemble_pipe1 = AutoFlowClassifier(
            resource_manager=self.mock_resource_manager
        )
        data_manager = deepcopy(data_manager)
        data_manager.resource_manager = ensemble_pipe1.resource_manager
        assert data_manager.X_train is None
        ensemble_pipe1.data_manager = data_manager
        ensemble_pipe1.estimator = ensemble_pipe1.fit_ensemble(
            task_id=pipe.task_id,
            trials_fetcher="GetSpecificTrials",
            trials_fetcher_params={"trial_ids": [0, 1, 2, 3, 4]}
        )
        assert ensemble_pipe1.experiment_id == 2
        score = ensemble_pipe1.score(X_test, y_test)
        assert score > 0.8
        assert len(ensemble_pipe1.estimator.estimators_list) == 4
        #######################################################
        ensemble_pipe2 = AutoFlowClassifier(
            resource_manager=self.mock_resource_manager
        )
        data_manager = deepcopy(data_manager)
        data_manager.resource_manager = ensemble_pipe2.resource_manager
        assert data_manager.X_train is None
        ensemble_pipe2.data_manager = data_manager
        ensemble_pipe2.estimator = ensemble_pipe2.fit_ensemble(
            task_id=pipe.task_id,
            trials_fetcher="GetBestK",
            trials_fetcher_params={"k": 5}
        )
        assert ensemble_pipe2.experiment_id == 3
        score = ensemble_pipe2.score(X_test, y_test)
        assert score > 0.8
        assert len(ensemble_pipe2.estimator.estimators_list) == 5
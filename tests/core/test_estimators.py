#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from autoflow.core.classifier import AutoFlowClassifier
from autoflow.tests.base import LocalResourceTestCase


class TestEstimators(LocalResourceTestCase):
    def test_single_algorithm(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        pipe = AutoFlowClassifier(
            DAG_workflow={
                "num->target": ["liblinear_svc", "libsvm_svc", "logistic_regression"]
            },
            initial_runs=1,
            run_limit=1,
            debug=True
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        # score = accuracy_score(y_test, y_pred)
        score = pipe.score(X_test, y_test)
        print(score)
        self.assertGreater(score, 0.9)

    def test_single_algorithm_with_X_test(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        pipe = AutoFlowClassifier(
            DAG_workflow={
                "num->target": ["liblinear_svc", "libsvm_svc", "logistic_regression"]
            },
            initial_runs=1,
            run_limit=1,
            debug=True,
            resource_manager=self.mock_resource_manager
        )
        pipe.fit(X_train, y_train, X_test, y_test)
        y_pred = pipe.predict(X_test)
        # score = accuracy_score(y_test, y_pred)
        score = pipe.score(X_test, y_test)
        print(score)
        self.assertGreater(score, 0.9)
        trial = pipe.resource_manager.TrialsModel
        records = trial.select().where(trial.experiment_id == pipe.experiment_id)
        for record in records:
            self.assertTrue(record is not None)
            self.assertTrue(
                isinstance(record.test_all_score, dict) and bool(record.test_all_score) and
                record.test_all_score["accuracy"] > 0.9)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold

from autoflow import AutoFlowClassifier
from autoflow.tests.base import LocalResourceTestCase


class TestSplitter(LocalResourceTestCase):
    def test_splitter(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        groups = np.zeros([X_train.shape[0]])
        groups[:len(groups) // 2] = 1
        pipe = AutoFlowClassifier(
            DAG_workflow={
                "num->target": ["liblinear_svc", "libsvm_svc", "logistic_regression"]
            },
            initial_runs=1,
            run_limit=1,
            debug=True,
            resource_manager=self.mock_resource_manager
        )
        pipe.fit(X_train, y_train, groups=groups)
        score = pipe.score(X_test, y_test)
        print(score)
        y_true_indexes = pipe.estimator.y_true_indexes_list[0]
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for observed_y_true_index, (_, valid_index) in zip(y_true_indexes, splitter.split(X_train, y_train, groups)):
            assert np.all(observed_y_true_index == valid_index)
        self.assertGreater(score, 0.5)

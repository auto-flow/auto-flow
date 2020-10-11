#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

from sklearn.model_selection import KFold

from autoflow import AutoFlowClassifier
from autoflow import datasets
from autoflow.tests.base import LocalResourceTestCase


class TestValidateHDL(LocalResourceTestCase):
    def test_validate_hdl1(self):
        train_df, test_df = datasets.load("titanic", return_train_test=True)
        trained_pipeline = AutoFlowClassifier(
            initial_runs=1, run_limit=3, n_jobs=1,
            included_classifiers=["dogboost"],
            debug=True,
            n_jobs_in_algorithm=5,
            resource_manager=self.mock_resource_manager
        )
        column_descriptions = {
            "id": "PassengerId",
            "target": "Survived",
            "text": "Name"
        }
        try:
            trained_pipeline.fit(
                X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
                splitter=KFold(n_splits=3, shuffle=True, random_state=42),
                fit_ensemble_params=True,
                is_not_realy_run=True
            )
            assert Exception("didn't detect wrong HDL.")
        except Exception as e:
            self.assertEqual(str(e), "In step 'final->target', user defined packege : 'dogboost' does not exist!")

    def test_validate_hdl2(self):
        train_df, test_df = datasets.load("titanic", return_train_test=True)
        trained_pipeline = AutoFlowClassifier(
            initial_runs=1, run_limit=3, n_jobs=1,
            included_highR_nan_imputers=["operate.pop"],
            debug=True,
            n_jobs_in_algorithm=5,
            resource_manager=self.mock_resource_manager
        )
        column_descriptions = {
            "id": "PassengerId",
            "target": "Survived",
            "text": "Name"
        }
        try:
            trained_pipeline.fit(
                X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
                splitter=KFold(n_splits=3, shuffle=True, random_state=42),
                fit_ensemble_params=True,
                is_not_realy_run=True
            )
            assert Exception("didn't detect wrong HDL.")
        except Exception as e:
            self.assertEqual(str(e), "In step 'highR_nan->nan', user defined packege : 'operate.pop' does not exist!")

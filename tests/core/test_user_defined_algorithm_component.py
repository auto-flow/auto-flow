#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier

from autoflow import AutoFlowClassifier
from autoflow.hdl.utils import get_default_hdl_bank
from autoflow.tests.base import LocalResourceTestCase
from autoflow.workflow.components.classification import AutoFlowClassificationAlgorithm


class MLPClassifier(AutoFlowClassificationAlgorithm):
    def fit(self, X_train, y_train=None,
            X_valid=None, y_valid=None,
            X_test=None, y_test=None):
        self.estimator = SklearnMLPClassifier(**self.hyperparams)
        self.estimator.fit(X_train.data, y_train.data)
        return self

    def predict(self, X):
        return self.estimator.predict(X.data)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X.data)

class TestUserDefinedAlgorithmComponent(LocalResourceTestCase):
    def test_classifier(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        hdl_bank = get_default_hdl_bank()
        hdl_bank["classification"]["mlp"] = {
            "hidden_layer_sizes": {"_type": "int_quniform", "_value": [10, 1000, 10]},
            "activation": {"_type": "choice", "_value": ["relu", "tanh", "logistic"]},
        }

        trained_pipeline = AutoFlowClassifier(
            initial_runs=1, run_limit=2, n_jobs=1,
            included_classifiers=["mlp"],
            hdl_bank=hdl_bank,
            model_registry={
                "mlp": MLPClassifier
            },
            debug=True,
            random_state=55,
            resource_manager=self.mock_resource_manager
        )
        trained_pipeline.fit(
            X_train=X, y_train=y, X_test=X_test, y_test=y_test,
            splitter=KFold(n_splits=3, shuffle=True, random_state=42),
        )
        joblib.dump(trained_pipeline, "autoflow_classification.bz2")
        # ---
        predict_pipeline = joblib.load("autoflow_classification.bz2")
        score = predict_pipeline.score(X_test, y_test)
        print(score)
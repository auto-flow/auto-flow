#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

import joblib
import pandas as pd
from sklearn.model_selection import KFold

from autoflow import AutoFlowClassifier
from autoflow.hdl.utils import get_default_hdl_bank
from autoflow.workflow.components.classification import AutoFlowClassificationAlgorithm
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier


class MLPClassifier(AutoFlowClassificationAlgorithm):
    def fit(self, X_train, y_train=None,
            X_valid=None, y_valid=None,
            X_test=None, y_test=None):
        self.estimator = SklearnMLPClassifier(**self.hyperparams)
        self.estimator.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.estimator.predict(X.data)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X.data)


hdl_bank = get_default_hdl_bank()
hdl_bank["classification"]["mlp"] = {
    "hidden_layer_sizes": {"_type": "int_quniform", "_value": [10, 1000, 10]},
    "activation": {"_type": "choice", "_value": ["relu", "tanh", "logistic"]},
}
train_df = pd.read_csv("./data/train_classification.csv")
test_df = pd.read_csv("./data/test_classification.csv")
trained_pipeline = AutoFlowClassifier(
    initial_runs=5, run_limit=10, n_jobs=1,
    included_classifiers=["mlp"],
    should_store_intermediate_result=True,
    hdl_bank=hdl_bank,
    model_registry={
        "mlp": MLPClassifier
    },
    debug=True,
    random_state=55
)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}
trained_pipeline.fit(
    X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
    splitter=KFold(n_splits=3, shuffle=True, random_state=42),
)
joblib.dump(trained_pipeline, "autoflow_classification.bz2")
predict_pipeline = joblib.load("autoflow_classification.bz2")
result = predict_pipeline.predict(test_df)
print(result)

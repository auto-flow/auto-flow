#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from autoflow.core.classifier import AutoFlowClassifier
from autoflow.data_container import DataFrameContainer
from autoflow.data_container import NdArrayContainer

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_test_ = DataFrameContainer(dataset_instance=X_test)
y_test_ = NdArrayContainer(dataset_instance=y_test)
pipe = AutoFlowClassifier()
estimator = pipe.fit_ensemble(
    task_id="2435e32babd7d09b6357e99aa7fa3b89",
    hdl_id="f289af8e23544a108bba6c8bc99673c3",
    trials_fetcher_params={"k": 50}
)
# pipe.fit(X_train, y_train, fit_ensemble_params=False)
# score = accuracy_score(y_test, y_pred)
y_pred = estimator.predict(X_test_)
score = accuracy_score(y_test, y_pred)
print(score)

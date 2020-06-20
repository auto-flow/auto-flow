#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from autoflow.core.classifier import AutoFlowClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe = AutoFlowClassifier(
    DAG_workflow={
        "num->target": [
            # "liblinear_svc", "libsvm_svc", "logistic_regression"
            # "random_forest",
            "logistic_regression",
            # "logistic_regression"
        ]
    },
    random_state=0,
    min_n_samples_for_SH=50,
    concurrent_type="thread",
    max_budget=1,
    min_budget=1 / 4
)
pipe.fit(X_train, y_train, fit_ensemble_params=False)
# score = accuracy_score(y_test, y_pred)
score = pipe.score(X_test, y_test)
print(score)

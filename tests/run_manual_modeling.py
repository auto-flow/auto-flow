#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from autoflow.core.classifier import AutoFlowClassifier

X, y = load_iris(return_X_y=True)
# X = X[y != 2]
# y = y[y != 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe = AutoFlowClassifier(
    DAG_workflow={
        "num->scaled": {
            "_name": "scale.standardize",
            "_vanilla":True
        },
        "scaled->target": {
            "_name": "liblinear_svc",
            "random_state":42,
            "_vanilla": True
        }
    },
    initial_runs=3,
    run_limit=9,
    n_jobs=3,
    debug=True,
    search_method="smac",
    random_state=0
)
pipe.fit(X_train, y_train, fit_ensemble_params=False)
# score = accuracy_score(y_test, y_pred)
score = pipe.score(X_test, y_test)
print(score)

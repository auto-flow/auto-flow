#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from autoflow.core.classifier import AutoFlowClassifier
from autoflow.resource_manager.http import HttpResourceManager

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
http_resource_manager = HttpResourceManager(db_params={
    "http_client": True,
    "url": "http://127.0.0.1:8000",
    "headers": {
        'Content-Type': 'application/json',
        'accept': 'application/json',
    }
})
pipe = AutoFlowClassifier(
    DAG_workflow={
        "num->target": ["liblinear_svc", "libsvm_svc", "logistic_regression"]
    },
    initial_runs=3,
    run_limit=9,
    n_jobs=3,
    debug=True,
    resource_manager=http_resource_manager
)
pipe.fit(
    X_train, y_train,
    fit_ensemble_params=False
)
# score = accuracy_score(y_test, y_pred)
score = pipe.score(X_test, y_test)
print(score)

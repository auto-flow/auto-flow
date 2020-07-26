#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from autoflow import HDL_Constructor
from autoflow.core.classifier import AutoFlowClassifier
from autoflow.ensemble.vote.classifier import VoteClassifier
from autoflow.resource_manager.http import HttpResourceManager
from autoflow.tuner import Tuner

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
hdl_constructors = [
    HDL_Constructor(
        DAG_workflow={
            "num->target": ["linearsvc", "svc", "logistic_regression"]
        },
    )
]*2
tuners = [
    Tuner(
        search_method="random",
        run_limit=3,
        n_jobs=3,
        debug=True
    ),
    Tuner(
        search_method="smac",
        initial_runs=3,
        run_limit=6,
        n_jobs=3,
        debug=True
    )
]
pipe = AutoFlowClassifier(
    hdl_constructor=hdl_constructors,
    tuner=tuners,
    resource_manager=http_resource_manager
)
pipe.fit(
    X_train, y_train,
    # fit_ensemble_params="auto",
    fit_ensemble_params=False,
)
assert isinstance(pipe.estimator, VoteClassifier)
# score = accuracy_score(y_test, y_pred)
score = pipe.score(X_test, y_test)
assert score > 0.8

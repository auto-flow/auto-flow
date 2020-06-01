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

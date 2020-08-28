#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from sklearn.datasets import load_iris

from autoflow import AutoFlowClassifier

autoflow = AutoFlowClassifier(
    evaluation_strategy="SH-5CV",
    n_iterations=1,
    warm_start=False,
    store_dataset=False,
    db_params={
        "user": "tqc",
        "host": "127.0.0.1",
        "port": 5432,
    },
    db_type="postgresql"
)
X, y = load_iris(True)

autoflow.fit(X, y, fit_ensemble_params=False)
print(autoflow)

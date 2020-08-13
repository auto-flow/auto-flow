#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import pandas as pd
from sklearn.datasets import load_digits, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from autoflow.feature_engineer.reduce import GenericDimentionReducer
from sklearn.pipeline import Pipeline

X, y = load_digits(return_X_y=True)
# X, y = load_iris(return_X_y=True)
cols = [f"#{i}" for i in range(X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train = pd.DataFrame(X_train, columns=cols)
X_test = pd.DataFrame(X_test, columns=cols)
pipe= Pipeline([  # lda pca
    ("reducer", GenericDimentionReducer(method="nmf", n_components="auto")),
    ("lr", LogisticRegression(random_state=42))
])
pipe.fit(X_train, y_train)
score=pipe.score(X_test, y_test)
print(score)
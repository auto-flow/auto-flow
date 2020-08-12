#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from autoflow.feature_engineer.generate.autofeat import AutoFeatGenerator

X, y = load_digits(return_X_y=True)
cols = [f"#{i}" for i in range(64)]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train = pd.DataFrame(X_train, columns=cols)
X_test = pd.DataFrame(X_test, columns=cols)
clf = AutoFeatGenerator(
    transformations=["1/"], verbose=1, max_used_feats=10, featsel_runs=3,
    consider_other=True, n_jobs=3
)

clf.fit(X_train, y_train)
X_train_ = clf.transform(X_train)
X_test_ = clf.transform(X_test)
lr = LogisticRegression(random_state=42).fit(X_train_, y_train)
score = lr.score(X_test_, y_test)
print(score)

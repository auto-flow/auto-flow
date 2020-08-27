#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from sklearn.datasets import load_iris

from autoflow import AutoFlowClassifier

autoflow = AutoFlowClassifier(evaluation_strategy="auto")
X, y = load_iris(True)

autoflow.fit(X, y)

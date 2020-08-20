#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.feature_engineer.text.topic.base import BaseSklearnTextTransformer


class TsvdTransformer(BaseSklearnTextTransformer):
    def __init__(self, num_topics=2, algorithm="randomized", n_iter=5,
                 random_state=42, tol=0.):
        self.tol = tol
        self.random_state = random_state
        self.n_iter = n_iter
        self.algorithm = algorithm
        self.num_topics = num_topics
        self.transformer_package = "sklearn.decomposition.TruncatedSVD"

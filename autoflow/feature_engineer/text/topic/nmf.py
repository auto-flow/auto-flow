#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.feature_engineer.text.topic.base import BaseSklearnTextTransformer


class NmfTransformer(BaseSklearnTextTransformer):
    def __init__(self, num_topics=None, init=None, solver='cd',
                 beta_loss='frobenius', tol=1e-4, max_iter=200,
                 random_state=None, alpha=0., l1_ratio=0., verbose=0,
                 shuffle=False):
        self.shuffle = shuffle
        self.verbose = verbose
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.beta_loss = beta_loss
        self.solver = solver
        self.init = init
        self.num_topics = num_topics
        self.transformer_package = "sklearn.decomposition.NMF"

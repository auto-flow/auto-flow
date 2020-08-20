#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

from autoflow.feature_engineer.text.topic.base import BaseGensim


class HdpTransformer(BaseGensim):
    def __init__(self, max_chunks=None, max_time=None, chunksize=256, kappa=1.0, tau=64.0, K=15, T=150,
                 alpha=1, gamma=1, eta=0.01, scale=1.0, var_converge=0.0001, outputdir=None, random_state=42):
        self.random_state = random_state
        self.outputdir = outputdir
        self.var_converge = var_converge
        self.scale = scale
        self.eta = eta
        self.gamma = gamma
        self.alpha = alpha
        self.T = T
        self.K = K
        self.tau = tau
        self.kappa = kappa
        self.chunksize = chunksize
        self.max_time = max_time
        self.max_chunks = max_chunks
        self.transformer_package = "gensim.sklearn_api.HdpTransformer"

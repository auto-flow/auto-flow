#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.feature_engineer.text.topic.base import BaseGensim


class LsiTransformer(BaseGensim):
    def __init__(self, num_topics=200, chunksize=20000,
                 decay=1.0, onepass=True, power_iters=2, extra_samples=100):
        self.extra_samples = extra_samples
        self.power_iters = power_iters
        self.onepass = onepass
        self.decay = decay
        self.chunksize = chunksize
        self.num_topics = num_topics
        self.transformer_package = "gensim.sklearn_api.LsiTransformer"

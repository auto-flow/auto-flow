#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.feature_engineer.text.topic.base import BaseGensim
import numpy as np


class LdaTransformer(BaseGensim):
    def __init__(self, num_topics=100,  chunksize=2000, passes=1, update_every=1, alpha='symmetric',
                 eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001,
                 minimum_probability=0.01, random_state=None, scorer='perplexity', dtype=np.float32):
        self.dtype = dtype
        self.scorer = scorer
        self.random_state = random_state
        self.minimum_probability = minimum_probability
        self.gamma_threshold = gamma_threshold
        self.iterations = iterations
        self.eval_every = eval_every
        self.offset = offset
        self.decay = decay
        self.eta = eta
        self.alpha = alpha
        self.update_every = update_every
        self.passes = passes
        self.chunksize = chunksize
        self.num_topics = num_topics
        self.transformer_package="gensim.sklearn_api.LdaTransformer"

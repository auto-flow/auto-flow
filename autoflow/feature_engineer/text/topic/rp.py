#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.feature_engineer.text.topic.base import BaseGensim

class RpTransformer(BaseGensim):
    def __init__(self,  num_topics=300):
        self.num_topics = num_topics
        self.transformer_package = "gensim.sklearn_api.RpTransformer"

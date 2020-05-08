#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com


from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["LdaTransformer"]


class LdaTransformer(AutoFlowFeatureEngineerAlgorithm):
    module__ = "autoflow.feature_engineer.text.topic"
    class__ = "LdaTransformer"
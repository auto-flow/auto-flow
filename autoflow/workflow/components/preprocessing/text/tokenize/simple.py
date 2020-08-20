#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com


from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["SimpleTokenlizer"]


class SimpleTokenlizer(AutoFlowFeatureEngineerAlgorithm):
    module__ = "autoflow.feature_engineer.text.tokenize"
    class__ = "SimpleTokenizer"

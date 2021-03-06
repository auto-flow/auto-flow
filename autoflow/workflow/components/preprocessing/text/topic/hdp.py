#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com


from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["HdpTransformer"]


class HdpTransformer(AutoFlowFeatureEngineerAlgorithm):
    module__ = "autoflow.feature_engineer.text.topic"
    class__ = "HdpTransformer"

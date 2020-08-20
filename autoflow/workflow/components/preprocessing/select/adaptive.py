#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["AdaptiveFeatureSelector"]


class AdaptiveFeatureSelector(AutoFlowFeatureEngineerAlgorithm):
    class__ = "AdaptiveFeatureSelector"
    module__ = "autoflow.feature_engineer.select"
    need_y = True

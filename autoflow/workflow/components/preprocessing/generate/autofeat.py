#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["AutoFeatureGenerator"]


class AutoFeatureGenerator(AutoFlowFeatureEngineerAlgorithm):
    class__ = "AutoFeatureGenerator"
    module__ = "autoflow.feature_engineer.generate"
    need_y = True
    cache_intermediate = True

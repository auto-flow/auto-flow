#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["BorutaFeatureSelector"]


class BorutaFeatureSelector(AutoFlowFeatureEngineerAlgorithm):
    class__ = "BorutaFeatureSelector"
    module__ = "autoflow.feature_engineer.select"
    need_y = True
    cache_intermediate = True

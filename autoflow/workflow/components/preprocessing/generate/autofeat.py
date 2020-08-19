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
    additional_info_keys = ("new_feat_cols_",)

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None, **kwargs):
        return estimator.fit(X, y, X_pool=[X_valid, X_test])

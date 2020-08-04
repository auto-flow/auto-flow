#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["AdaptiveImputer"]


class AdaptiveImputer(AutoFlowFeatureEngineerAlgorithm):
    class__ = "AdaptiveImputer"
    module__ = "autoflow.feature_engineer.impute"

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None, **kwargs):
        columns = X.columns
        cat_cols = columns[feature_groups.isin(["cat", "highC_cat"])]
        num_cols = columns[feature_groups.isin(["num"])]
        return estimator.fit(X, y, categorical_feature=cat_cols, numerical_feature=num_cols)

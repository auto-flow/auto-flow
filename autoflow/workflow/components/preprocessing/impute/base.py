#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["BaseImputer"]


class BaseImputer(AutoFlowFeatureEngineerAlgorithm):
    class__ = "BaseImputer"
    module__ = "autoflow.feature_engineer.impute"
    cache_intermediate = True
    additional_info_keys = ("missing_rates", "drop_columns")

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None, **kwargs):
        columns = X.columns
        cat_cols = columns[feature_groups.isin(["cat", "highC_cat"])]
        num_cols = columns[feature_groups.isin(["num"])]
        return estimator.fit(X, y, categorical_feature=cat_cols, numerical_feature=num_cols)

    def assemble_all_result(self, X_stack, X_trans, X_train, X_valid, X_test, y_train):
        X_stack.feature_groups=X_stack.feature_groups[~self.component.drop_mask].reset_index(drop=True)
        return super(BaseImputer, self).assemble_all_result( X_stack, X_trans, X_train, X_valid, X_test, y_train)

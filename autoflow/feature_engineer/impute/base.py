#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import category_encoders.utils as util
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from autoflow.utils.logging_ import get_logger


class BaseImputer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            categorical_feature=None,
            numerical_feature=None,
            copy=True
    ):
        self.numerical_feature = numerical_feature
        self.copy = copy
        self.categorical_feature = categorical_feature
        self.logger = get_logger(self)

    def fit(self, X, y=None, categorical_feature=None, numerical_feature=None, **kwargs):
        X = util.convert_input(X)
        if categorical_feature is not None:
            self.categorical_feature = categorical_feature
        if numerical_feature is not None:
            self.numerical_feature = numerical_feature
        # todo: 自动找category特征
        if self.categorical_feature is None and self.numerical_feature is None:
            self.logger.warning(
                f"You didn't declare numerical_feature or categorical_feature in {self.__class__.__name__}, "
                f"program will auto find these by dtypes.")
            self.categorical_feature = X.select_dtypes(include=["object", "category"]).columns
            self.numerical_feature = X.select_dtypes(exclude=["object", "category"]).columns
        else:
            if self.categorical_feature is None:
                if self.numerical_feature is not None:
                    self.categorical_feature = X.columns.difference(self.numerical_feature)
                else:
                    self.categorical_feature = np.array([])
            if numerical_feature is None:
                if self.categorical_feature is not None:
                    self.numerical_feature = X.columns.difference(self.categorical_feature)
                else:
                    self.numerical_feature = np.array([])
        return X

    def transform(self, X):
        X = util.convert_input(X)
        if self.copy:
            X = X.copy()
        return X



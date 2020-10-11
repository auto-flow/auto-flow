#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import category_encoders.utils as util
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from autoflow.utils.logging_ import get_logger


class BaseImputer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            categorical_feature=None,
            numerical_feature=None,
            copy=True,
            missing_rate=0.4,
            inclusive=True,
    ):
        self.inclusive = inclusive
        self.missing_rate = missing_rate
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
        #  自动找category特征
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
        # todo: 统计各列的缺失率，过高则删除
        missing_rates = np.count_nonzero(pd.isna(X), axis=0) / X.shape[0]
        self.missing_rates = missing_rates.tolist()
        if self.inclusive:
            drop_mask = missing_rates >= self.missing_rate
        else:
            drop_mask = missing_rates > self.missing_rate
        self.drop_mask = drop_mask
        drop_columns = X.columns[drop_mask]
        self.drop_columns = drop_columns.tolist()
        if len(drop_columns):
            self.numerical_feature = np.setdiff1d(self.numerical_feature, drop_columns)
            self.categorical_feature = np.setdiff1d(self.categorical_feature, drop_columns)
            X = X.drop(drop_columns, axis=1)
        return X

    def transform(self, X):
        X = util.convert_input(X)
        if len(self.drop_columns):
            X = X.drop(self.drop_columns, axis=1)
        if self.copy:
            X = X.copy()
        return X

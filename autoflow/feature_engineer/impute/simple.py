#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from collections import Counter
from copy import copy

import category_encoders.utils as util
import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer as SklearnSimpleImputer
from sklearn.utils._testing import ignore_warnings


class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            strategy="most_frequent",
            fill_value="<NULL>",
    ):
        self.strategy = strategy
        self.fill_value = fill_value
        assert strategy in ("most_frequent", "constant"), ValueError(f"Invalid strategy {strategy}")

    def fit(self, X, y=None):
        X = util.convert_input(X)
        self.columns = X.columns
        self.statistics_ = [self.fill_value] * len(self.columns)
        if self.strategy == "most_frequent":
            for i, column in enumerate(X.columns):
                for value, counts in Counter(X[column]).items():
                    if not pd.isna(value):
                        self.statistics_[i] = value
                        continue
        return self

    @ignore_warnings(category=SettingWithCopyWarning)
    def transform(self, X):
        # note: change inplace
        for i, (column, dtype) in enumerate(zip(X.columns, X.dtypes)):
            value = self.statistics_[i]
            mask = pd.isna(X[column]).values
            if np.count_nonzero(mask) == 0:
                continue
            if dtype.name == "category" and value not in X[column].cat.categories:
                X[column].cat.add_categories(value, inplace=True)
            X.loc[mask,column ] = value
        return X


class SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            categorical_feature=None,
            numerical_feature=None,
            num_strategy="median",
            cat_strategy="most_frequent",
            copy=True
    ):
        self.numerical_feature = numerical_feature
        self.copy = copy
        self.categorical_feature = categorical_feature
        assert num_strategy in ("median", "mean")
        # fill_value = None
        # if cat_strategy != "most_frequent":
        #     fill_value = cat_strategy
        #     cat_strategy = "constant"
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        # self.fill_value = fill_value

    def fit(self, X, y=None, categorical_feature=None, numerical_feature=None, **kwargs):
        X = util.convert_input(X)
        if categorical_feature is not None:
            self.categorical_feature = categorical_feature
        if numerical_feature is not None:
            self.numerical_feature = numerical_feature
        cat_cols = self.categorical_feature
        num_cols = self.numerical_feature
        if cat_cols is None:
            cat_cols = []
        cat_cols = np.array(cat_cols)
        columns = np.array(X.columns)
        if num_cols is None:
            num_cols = np.setdiff1d(columns, cat_cols)
        else:
            num_cols = np.array([])
        if num_cols.size:
            self.num_imputer = SklearnSimpleImputer(strategy=self.num_strategy).fit(X[num_cols])
        else:
            self.num_imputer = None
        if cat_cols.size:
            self.cat_imputer = CategoricalImputer(strategy=self.cat_strategy).fit(X[cat_cols])
        else:
            self.cat_imputer = None
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        return self

    def transform(self, X):
        X = util.convert_input(X)
        if self.copy:
            X = copy(X)
        if self.cat_imputer is not None:
            X[self.cat_cols] = self.cat_imputer.transform(X[self.cat_cols])
        if self.num_imputer is not None:
            X[self.num_cols] = self.num_imputer.transform(X[self.num_cols])
        return X

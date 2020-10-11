#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from collections import Counter

import category_encoders.utils as util
import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer as SklearnSimpleImputer
from sklearn.utils._testing import ignore_warnings

from .base import BaseImputer


class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            strategy="most_frequent",
            fill_value="<NULL>",
            numeric_fill_value=-1
    ):
        self.numeric_fill_value = numeric_fill_value
        self.strategy = strategy
        self.fill_value = fill_value
        assert strategy in ("most_frequent", "constant"), ValueError(f"Invalid strategy {strategy}")

    def fit(self, X, y=None):
        X = util.convert_input(X)
        self.columns = X.columns
        self.statistics_ = np.array([self.fill_value] * len(self.columns), dtype='object')
        numeric_mask = np.array([is_numeric_dtype(dtype) for dtype in X.dtypes])
        self.statistics_[numeric_mask] = self.numeric_fill_value
        if self.strategy == "most_frequent":
            for i, column in enumerate(X.columns):
                for value, counts in Counter(X[column]).most_common():
                    if not pd.isna(value):
                        self.statistics_[i] = value
                        break
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
            X.loc[mask, column] = value
        return X


class SimpleImputer(BaseImputer):
    def __init__(
            self,
            categorical_feature=None,
            numerical_feature=None,
            copy=True,
            missing_rate=0.4,
            num_strategy="median",
            cat_strategy="most_frequent",
            inclusive=True,
    ):
        super(SimpleImputer, self).__init__(
            categorical_feature,
            numerical_feature,
            copy,
            missing_rate,
            inclusive=inclusive
        )
        assert num_strategy in ("median", "mean")
        assert cat_strategy in ("most_frequent", "constant")
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy

    def fit(self, X, y=None, categorical_feature=None, numerical_feature=None, **kwargs):
        X = super(SimpleImputer, self).fit(X, y, categorical_feature, numerical_feature)
        cat_cols = self.categorical_feature
        num_cols = self.numerical_feature
        if num_cols.size:
            self.num_imputer = SklearnSimpleImputer(strategy=self.num_strategy) #.fit(X[num_cols])
        else:
            self.num_imputer = None
        if cat_cols.size:
            self.cat_imputer = CategoricalImputer(strategy=self.cat_strategy) #.fit(X[cat_cols])
        else:
            self.cat_imputer = None
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        return self

    def transform(self, X):
        X = super(SimpleImputer, self).transform(X)
        X.loc[:, pd.isna(X).sum() == X.shape[0]] = 0
        if self.cat_imputer is not None:
            X[self.cat_cols] = self.cat_imputer.fit_transform(X[self.cat_cols])
        if self.num_imputer is not None:
            X[self.num_cols] = self.num_imputer.fit_transform(X[self.num_cols])
        return X

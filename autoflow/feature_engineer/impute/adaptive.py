#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy

import category_encoders.utils as util
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer


class AdaptiveImputer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            categorical_feature=None,
            num_strategy="median",
            cat_strategy="most_frequent",
            copy=True
    ):
        self.copy = copy
        self.categorical_feature = categorical_feature
        assert num_strategy in ("median", "mean")
        fill_value = None
        if cat_strategy != "most_frequent":
            fill_value = cat_strategy
            cat_strategy = "constant"
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.fill_value = fill_value

    def fit(self, X, y=None, categorical_feature=None, **kwargs):
        X = util.convert_input(X)
        if categorical_feature is not None:
            self.categorical_feature = categorical_feature
        cat_cols = self.categorical_feature
        if cat_cols is None:
            cat_cols = []
        cat_cols = np.array(cat_cols)
        columns = np.array(X.columns)
        num_cols = np.setdiff1d(columns, cat_cols)

        if num_cols.size:
            self.num_imputer = SimpleImputer(strategy=self.num_strategy).fit(X[num_cols])
        else:
            self.num_imputer = None
        if cat_cols.size:
            self.cat_imputer = SimpleImputer(strategy=self.cat_strategy, fill_value=self.fill_value). \
                fit(X[cat_cols])
        else:
            self.cat_imputer = None
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        return self

    def transform(self, X):
        X = util.convert_input(X)
        if self.copy:
            X = deepcopy(X)
        if self.cat_imputer is not None:
            X[self.cat_cols] = self.cat_imputer.transform(X[self.cat_cols])
        if self.num_imputer is not None:
            X[self.num_cols] = self.num_imputer.transform(X[self.num_cols])
        return X

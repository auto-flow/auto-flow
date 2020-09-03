#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from collections import Counter
from copy import deepcopy

import numpy as np
from category_encoders.utils import convert_input
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["CombineRare"]


class CombineRare(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            minimum_fraction=0.1,
            rare_category="Others Infrequent",
            rare_category_numeric=999,
            copy=True,
            drop_invariant=True,
            categorical_feature=None
    ):
        self.categorical_feature = categorical_feature
        self.rare_category_numeric = rare_category_numeric
        self.drop_invariant = drop_invariant
        self.copy = copy
        self.rare_category = rare_category
        self.minimum_fraction = minimum_fraction
        self.invariant_cols = []
        self.replace_by_other = []

    def fit(self, X, y=None, **kwargs):
        X = convert_input(X)  # todo drop var = 0
        do_not_replace_by_other: Dict[str, list] = {}
        # 遍历每列
        if self.categorical_feature is None:
            self.categorical_feature = X.columns
        for column in self.categorical_feature:
            if column not in X.columns:
                continue
            do_not_replace_by_other[column] = []
            counter = Counter(X[column])
            colsize = X.shape[0]
            if X[column].dtype.name == "category":
                categories = list(X[column].cat.categories)
            else:
                categories = list(set(X[column]))
            for unique_value in categories:
                count = counter[unique_value]
                minimum_fraction = float(count) / colsize
                if minimum_fraction >= self.minimum_fraction:
                    do_not_replace_by_other[column].append(unique_value)
        self.do_not_replace_by_other_ = do_not_replace_by_other
        return self

    def transform(self, X):
        X = convert_input(X)
        if self.copy:
            X = deepcopy(X)
        self.invariant_cols = []
        self.replace_by_other = []
        for column in self.categorical_feature:
            if column not in X.columns:
                continue
            dtype = X[column].dtype
            if is_numeric_dtype(dtype):
                rare_category = self.rare_category_numeric
            else:
                rare_category = self.rare_category
            is_cat_dtype = True if dtype.name == "category" else False
            valid_cats = self.do_not_replace_by_other_[column]
            if len(valid_cats) == 0:
                if self.drop_invariant:
                    self.invariant_cols.append(column)
                    continue
            # todo: keep origin category order
            rare_mask = ~X[column].isin(valid_cats).values
            rare_values = list(set(np.array(X[column]).flatten()[rare_mask].tolist()))
            self.replace_by_other.append(rare_values)
            if not np.any(rare_mask):
                continue
            if is_cat_dtype:
                # add rare_category avoid error
                X[column].cat.add_categories(rare_category, inplace=True)
            X.loc[rare_mask, column] = rare_category
            if is_cat_dtype:
                # reset category , only keep used categories
                new_cat = valid_cats + [rare_category]
                X[column].cat.set_categories(new_cat, inplace=True)
        if self.drop_invariant:
            X.drop(self.invariant_cols, axis=1, inplace=True)
        return X

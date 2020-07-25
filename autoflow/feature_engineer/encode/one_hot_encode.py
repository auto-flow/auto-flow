#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
from category_encoders import OneHotEncoder as OriginOneHotEncoder
from category_encoders.utils import convert_input
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["OneHotEncoder"]


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            minimum_fraction=None,
            verbose=0,
            cols=None,
            drop_invariant=False,
            return_df=True,
            handle_missing='value',
            handle_unknown='value',
            use_cat_names=False,
    ):
        self.minimum_fraction = minimum_fraction
        self.use_cat_names = use_cat_names
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.cols = cols
        self.verbose = verbose
        self.ohe = OriginOneHotEncoder(
            verbose=verbose,
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_missing=handle_missing,
            handle_unknown=handle_unknown,
            use_cat_names=use_cat_names
        )

    def fit(self, X, y=None, **kwargs):
        X = convert_input(X)
        # 遍历每列
        X_ = X.values
        do_not_replace_by_other = list()
        converted_values = list()
        if self.minimum_fraction is not None:
            for column in range(X.shape[1]):
                do_not_replace_by_other.append(set())
                unique, counts = np.unique(
                    X_[:, column], return_counts=True,
                )
                colsize = X.shape[0]
                for unique_value, count in zip(unique, counts):
                    fraction = float(count) / colsize
                    if fraction >= self.minimum_fraction:
                        do_not_replace_by_other[-1].add(unique_value)
                converted_value = None
                for unique_value in unique:
                    if unique_value not in do_not_replace_by_other[-1]:
                        if converted_value is None:
                            converted_value = unique_value
                        X_[:, column][(X_[:, column] == unique_value)] = converted_value
                converted_values.append(converted_value)
        self.do_not_replace_by_other_ = do_not_replace_by_other
        self.converted_values_ = converted_values
        self.ohe.fit(X.astype(str), y, **kwargs)
        return self

    def transform(self, X):
        X = convert_input(X)
        X_ = X.values
        if self.minimum_fraction is not None:
            for column in range(X_.shape[1]):
                unique = np.unique(X_[:, column])
                for unique_value in unique:
                    if unique_value not in self.do_not_replace_by_other_[column]:
                        X_[:, column][(X_[:, column] == unique_value)] = self.converted_values_[column]
        return self.ohe.transform(X.astype(str))

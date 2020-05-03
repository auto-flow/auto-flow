#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from collections import Counter
from functools import reduce, partial
from operator import add

from sklearn.base import BaseEstimator, TransformerMixin

from autoflow.utils.dataframe import process_dataframe


def clean_text(text):
    text = text.replace("\n", " ").replace("\r", " ")
    punc_list = '''!"'#$&()*+,-./:;<=>?@[\]^_{|}~`0123456789'''
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.translate(t)
    return text


def remove_low_requency(counter, items):
    return [item for item in items if item in counter]


class SimpleTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, min_frequency=1):
        self.min_frequency = max(0, min_frequency)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = process_dataframe(X)
        for column in X_.columns:
            col = X[column]
            col_clean = col.apply(clean_text)
            col_split = col_clean.str.split()

            if self.min_frequency > 0:
                flatten_items = reduce(add, col_split)
                counter = Counter(flatten_items)
                counter = {k: v for k, v in counter.items() if v > self.min_frequency}
                filter_func = partial(remove_low_requency, counter)
                col_final = col_split.apply(filter_func)
            else:
                col_final = col_split
            X_[column] = col_final
        return X_

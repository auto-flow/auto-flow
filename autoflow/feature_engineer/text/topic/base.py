#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

import pandas as pd
from gensim.corpora import Dictionary
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from autoflow.utils.dataframe import process_dataframe
from autoflow.utils.klass import gather_kwargs_from_signature_and_attributes
from autoflow.utils.packages import import_by_package_url


class BaseGensim(BaseEstimator, TransformerMixin):
    transformer_package = None

    def series2bows(self, column, series: pd.Series):
        dic = self.column2dict[column]
        bows = [dic.doc2bow(item) for item in series]
        return bows

    def fit(self, X, y=None):
        X = process_dataframe(X, False)
        self.column2dict = {column: Dictionary(X[column]) for column in X.columns}
        self.column2transformer = {}
        for column in X.columns:
            klass = import_by_package_url(self.transformer_package)
            kwargs = gather_kwargs_from_signature_and_attributes(klass, self)
            kwargs.update({"id2word": self.column2dict[column]})
            transformer = klass(**kwargs)
            bows = self.series2bows(column, X[column])
            self.column2transformer[column] = transformer.fit(bows)
        return self

    def transform(self, X):
        X = process_dataframe(X, False)
        results = []
        for column in X.columns:
            series = X[column]
            bows = self.series2bows(column, series)
            transformer = self.column2transformer[column]
            data = transformer.transform(bows)
            columns = [f"{column}_{i}" for i in range(data.shape[1])]
            result = pd.DataFrame(data, columns=columns, index=X.index)
            results.append(result)
        return pd.concat(results, axis=1)


class BaseSklearnTextTransformer(BaseEstimator, TransformerMixin):
    transformer_package = None
    N_COMPONENTS = "n_components"
    NUM_TOPICS = "num_topics"

    def fit(self, X, y=None):
        X = process_dataframe(X, False)
        self.column2vectorizer = {}
        self.column2transformer = {}
        for column in X.columns:
            series = X[column]
            str_series = series.str.join(" ")
            vectorizer = TfidfVectorizer()
            vectorizer.fit(str_series)
            self.column2vectorizer[column] = vectorizer
            tfidf = vectorizer.transform(str_series)
            klass = import_by_package_url(self.transformer_package)
            kwargs = gather_kwargs_from_signature_and_attributes(klass, self)
            if self.NUM_TOPICS in kwargs:
                kwargs[self.N_COMPONENTS] = kwargs.pop(self.NUM_TOPICS)
            transformer = klass(**kwargs)
            transformer.fit(tfidf)
            self.column2transformer[column] = transformer
        return self

    def transform(self, X):
        X = process_dataframe(X, False)
        results = []
        for column in X.columns:
            series = X[column]
            str_series = series.str.join(" ")
            vectorizer = self.column2vectorizer[column]
            transformer = self.column2transformer[column]
            data = transformer.transform(vectorizer.transform(str_series))
            columns = [f"{column}_{i}" for i in range(data.shape[1])]
            result = pd.DataFrame(data, columns=columns, index=X.index)
            results.append(result)
        return pd.concat(results, axis=1)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from time import time

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD, NMF, PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning, ChangedBehaviorWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import check_X_y
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.multiclass import type_of_target

from autoflow.utils.logging_ import get_logger


class GenericDimentionReducer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            method="tsvd",
            n_components="auto",
            problem_type=None,
            random_state=0,
            budget=10,
            n_jobs=-1
    ):
        self.budget = budget
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.problem_type = problem_type
        self.method = method
        self.logger = get_logger(self)

    def fit(self, X, y):
        if self.problem_type is None:
            if type_of_target(y) == "continuous":
                self.problem_type = "regression"
            else:
                self.problem_type = "classification"
        X, y = check_X_y(X, y, y_numeric=self.problem_type == "regression", dtype=None)
        assert not (self.method == "lda" and self.problem_type == "regression"), ValueError(
            f"method 'lda' can only be used in classification problem-type!")
        M = min(*X.shape)
        if self.method == "lda":
            M = len(set(y))
        if isinstance(self.n_components, float):
            fraction = self.n_components
            n_components = int(np.clip(round(M * (fraction)), 1, M - 1))
        elif isinstance(self.n_components, int):
            n_components = int(np.clip(self.n_components, 1, M - 1))
        elif isinstance(self.n_components, str) and self.n_components == "auto":
            n_components = self.auto_find_n_components(M, X, y)
        else:
            raise ValueError(f"Unknown n_components {repr(self.n_components)} !")
        self.parsed_n_components = n_components
        self.reducer = self.get_reducer(n_components)
        self.fit_reducer(self.reducer, X, y)
        return self

    @ignore_warnings(category=ConvergenceWarning)
    @ignore_warnings(category=ChangedBehaviorWarning)
    @ignore_warnings(category=FutureWarning)
    def evaluate(self, n_components, X, y):
        reducer = self.get_reducer(n_components)
        self.fit_reducer(reducer, X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state, test_size=0.25)
        pipeline = Pipeline([
            ("reducer", reducer),
            ("lr",
             LogisticRegression(random_state=self.random_state) if self.problem_type == "classification" else
             Ridge(random_state=self.random_state))
        ]).fit(X_train, y_train)
        return pipeline.score(X_test, y_test)

    def auto_find_n_components(self, M, X, y):
        self.N2score = {}
        self.step = 0
        if M >= 1000:
            return M // 20
        start_time = time()
        initial_Ns = [
            int(round(M / 3)),
            int(round(M * 0.67))
        ]
        initial_Ns = [np.clip(initial_N, 1, M - 1) for initial_N in initial_Ns]
        initial_Ns = list(set(initial_Ns))
        N2score = {initial_N: self.evaluate(initial_N, X, y) for initial_N in initial_Ns}
        if time() - start_time > self.budget:
            Ns = sorted(list(N2score.keys()))
            scores = [N2score[N] for N in Ns]
            best_N_ix = int(np.argmax(scores))
            self.step = 0
            self.N2score = N2score
            return Ns[best_N_ix]
        step = 0
        for step in range(10):
            Ns = sorted(list(N2score.keys()))
            scores = [N2score[N] for N in Ns]
            best_N_ix = int(np.argmax(scores))
            cand_Ns = []
            if best_N_ix == 0:
                cand_Ns.append(max(1, Ns[best_N_ix] // 2))
            else:
                cand_Ns.append(
                    int(round(Ns[best_N_ix - 1] + (Ns[best_N_ix] - Ns[best_N_ix - 1]) / 2))
                )
            if best_N_ix != len(Ns) - 1:
                cand_Ns.append(
                    int(round(Ns[best_N_ix] + (Ns[best_N_ix + 1] - Ns[best_N_ix]) / 2))
                )
            else:
                self.logger.debug(f"best_N_ix = {best_N_ix}, equals `len(Ns) - 1)`")
            cand_Ns = [np.clip(cand_N, 1, M - 1) for cand_N in cand_Ns]
            should_break = False
            for N in cand_Ns:
                if N in N2score:
                    should_break = True
                    break
                if N <= 0:
                    should_break = True
                    break
            if should_break:
                break
            should_break = False
            for cand_N in cand_Ns:
                N2score[cand_N] = self.evaluate(cand_N, X, y)
                if time() - start_time > self.budget:
                    should_break = True
                    break
            if should_break:
                break
        self.step = step
        Ns = sorted(list(N2score.keys()))
        scores = [N2score[N] for N in Ns]
        best_N_ix = int(np.argmax(scores))
        self.N2score = N2score
        return Ns[best_N_ix]

    def transform(self, X):
        return self.transform_reducer(self.reducer, X)

    def get_reducer(self, n_components):
        if self.method == "tsvd":
            reducer = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        elif self.method == "lda":
            reducer = LinearDiscriminantAnalysis(n_components=n_components)
        elif self.method == "pca":
            reducer = PCA(n_components=n_components, random_state=self.random_state)
        elif self.method == "nmf":
            reducer = NMF(n_components=n_components, random_state=self.random_state)
        elif self.method == "tsne":
            reducer = TSNE(n_components=n_components, n_jobs=self.n_jobs, random_state=self.random_state)
        elif self.method == "ica":
            reducer = FastICA(n_components=n_components, random_state=self.random_state)
        else:
            raise ValueError(f"Invalid method '{self.method}'")
        return reducer

    def fit_reducer(self, reducer, X, y):
        if self.method == "lda":
            reducer.fit(X, y)
        return reducer

    def transform_reducer(self, reducer, X):
        if self.method == "lda":
            return reducer.transform(X)
        else:
            return reducer.fit_transform(X)

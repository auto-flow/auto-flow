#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy

import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from autoflow.opt.config_generators.density_estimator.base import BaseDensityEstimator


class KDE4BO(BaseDensityEstimator):
    def __init__(
            self,
            top_n_percent=15, bandwidth_factor=3, min_bandwidth=1e3,
            bw_estimation="normal_reference", min_points_in_kde=2
    ):
        super(KDE4BO, self).__init__(
            top_n_percent, bandwidth_factor, min_bandwidth,
            bw_estimation, min_points_in_kde
        )
        self.good_kde = None
        self.bad_kde = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        super(KDE4BO, self).fit(X, y)
        self.kde_vartypes = "".join(["u" if n_choices > 0 else "c" for n_choices in self.n_choices_list])
        n_good = max(2, (self.top_n_percent * X.shape[0]) // 100)
        N = X.shape[0]
        L = len(self.n_choices_list)
        if n_good <= L or N - n_good <= L:
            return None
        idx = np.argsort(y)
        if self.good_kde is None:
            good_kde_bw = np.zeros([len(self.n_choices_list)]) + 0.1
            bad_kde_bw = deepcopy(good_kde_bw)
        else:
            good_kde_bw = self.good_kde.bw
            bad_kde_bw = self.bad_kde.bw
        X_good = X[idx[:n_good]]
        X_bad = X[idx[n_good:]]
        for X_, bw_vector in zip([X_good, X_bad], [good_kde_bw, bad_kde_bw]):
            M = X_.shape[1]
            for i in range(M):
                bw = bw_vector[i]
                n_choices = self.n_choices_list[i]
                X_[:, i] = self.process_constants_vector(X_[:, i], n_choices, bw, mode="replace")
        self.good_kde = KDEMultivariate(data=X_good, var_type=self.kde_vartypes, bw=self.bw_estimation)
        self.bad_kde = KDEMultivariate(data=X_bad, var_type=self.kde_vartypes, bw=self.bw_estimation)
        return self

    def predict(self, X: np.ndarray):
        super(KDE4BO, self).predict(X)
        good_pdf = self.good_kde.pdf(X)
        bad_pdf = self.bad_kde.pdf(X)
        return good_pdf / bad_pdf

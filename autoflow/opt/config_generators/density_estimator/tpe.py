#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import random
from copy import deepcopy
from typing import List, Optional

import numpy as np
from scipy.stats import truncnorm
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from autoflow.opt.config_generators.density_estimator.base import BaseDensityEstimator


class ChoicesKDE:
    def __init__(self, data):
        self.data = data
        self.prob = {}
        self.bw = 0.2
        for number, cnt in zip(*np.unique(data, return_counts=True)):
            self.prob[int(number)] = cnt / data.size

    def pdf(self, X):
        result = np.zeros_like(X, dtype="float64") + 1e-32
        for number in np.unique(X):
            result[X == number] = self.prob.get(int(number), 1e-32)
        return result


class TreeBasedParzenEstimator(BaseDensityEstimator):
    def __init__(
            self,
            top_n_percent=15, bandwidth_factor=3, min_bandwidth=1e3,
            bw_estimation="normal_reference", min_points_in_kde=2,
            fill_deactivated_value=False
    ):
        super(TreeBasedParzenEstimator, self).__init__(
            top_n_percent,bandwidth_factor,min_bandwidth,
            bw_estimation,min_points_in_kde
        )
        self.fill_deactivated_value = fill_deactivated_value
        self.good_kdes = None
        self.bad_kdes = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        super(TreeBasedParzenEstimator, self).fit(X,y)
        good_kdes = []
        bad_kdes = []
        for i in range(X.shape[1]):
            if self.good_kdes is not None and self.good_kdes[i] is not None:
                previous_good_bw = self.good_kdes[i].bw
            else:
                previous_good_bw = 0.5
            if self.bad_kdes is not None and self.bad_kdes[i] is not None:
                previous_bad_bw = self.bad_kdes[i].bw
            else:
                previous_bad_bw = 0.5
            n_choices = self.n_choices_list[i]
            X_ = X[:, i]
            mask = (~np.isnan(X_))
            activated_X = X_[mask]
            N = np.count_nonzero(mask)
            # Each KDE contains at least 2 samples,
            # two KDEs contains at least 4 samples
            if N <= 3:
                # Too few observation samples
                good_kdes.append(None)
                bad_kdes.append(None)
                continue
            # Each KDE contains at least 2 samples
            n_good = max(2, (self.top_n_percent * activated_X.shape[0]) // 100)
            if n_good < self.min_points_in_kde or \
                    N - n_good < self.min_points_in_kde:
                # Too few observation samples
                good_kdes.append(None)
                bad_kdes.append(None)
                continue
            activated_y = y[mask]
            idx = np.argsort(activated_y)
            activated_X_good = activated_X[idx[:n_good]]
            activated_X_bad = activated_X[idx[n_good:]]

            activated_X_good = self.process_constants_vector(activated_X_good, n_choices, previous_good_bw)
            activated_X_bad = self.process_constants_vector(activated_X_bad, n_choices, previous_bad_bw)
            activated_X_good = activated_X_good[:, None]
            activated_X_bad = activated_X_bad[:, None]
            if n_choices == 0:
                var_type = "c"  # continuous
                good_kdes.append(KDEMultivariate(activated_X_good, var_type, self.bw_estimation))
                bad_kdes.append(KDEMultivariate(activated_X_bad, var_type, self.bw_estimation))
            else:
                var_type = "u"  # unordered
                # good_kdes.append(KDEMultivariate(activated_X_good, var_type, self.bw_estimation))
                # bad_kdes.append(KDEMultivariate(activated_X_bad, var_type, self.bw_estimation))
                good_kdes.append(ChoicesKDE(activated_X_good))
                bad_kdes.append(ChoicesKDE(activated_X_bad))
        self.good_kdes = good_kdes
        self.bad_kdes = bad_kdes
        return self

    def predict(self, X: np.ndarray):
        super(TreeBasedParzenEstimator, self).predict(X)
        good_pdf = np.ones_like(X, dtype="float64")
        bad_pdf = deepcopy(good_pdf)
        for i, (good_kde, bad_kde) in enumerate(zip(self.good_kdes, self.bad_kdes)):
            if good_kde is None or bad_kde is None:
                continue
            X_ = X[:, i]
            mask = (~np.isnan(X_))
            activated_X = X_[mask]
            N = np.count_nonzero(mask)
            if N == 0:
                continue
            good_pdf_activated = good_kde.pdf(activated_X)
            bad_pdf_activated = bad_kde.pdf(activated_X)
            good_pdf[mask, i] = good_pdf_activated
            bad_pdf[mask, i] = bad_pdf_activated
            N_deactivated = np.count_nonzero(~mask)
            if N_deactivated > 0 and self.fill_deactivated_value:
                good_pdf[~mask, i] = np.random.choice(good_pdf_activated)
                bad_pdf[~mask, i] = np.random.choice(bad_pdf_activated)
        good_pdf[good_pdf == 0] = 1e-32
        bad_pdf[bad_pdf == 0] = 1e-32
        log_good_pdf = np.log(good_pdf)
        log_bad_pdf = np.log(bad_pdf)
        log_good_pdf[~np.isfinite(log_good_pdf)] = -10
        log_bad_pdf[log_bad_pdf == -np.inf] = -10
        log_bad_pdf[~np.isfinite(log_bad_pdf)] = 10
        result = log_good_pdf.sum(axis=1) - log_bad_pdf.sum(axis=1)
        return result

    def process_constants_vector(self, vec: np.ndarray, n_choices, bw):
        if len(set(vec)) == 1:
            if n_choices > 1:
                return vec
                # others = set(range(n_choices)) - set(vec)
                # other = random.choice(list(others))
            elif n_choices == 0:
                m = vec[0]
                bw=max(0.1,bw)
                while True:
                    other = truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw)
                    other = np.clip(other, 0, 1)
                    if other!=m:
                        break
            else:
                raise ValueError
            return np.hstack([vec, [other]])
        else:
            return vec



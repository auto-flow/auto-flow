#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import random
from typing import List, Optional

import numpy as np
from scipy.stats import truncnorm
from sklearn.base import BaseEstimator

from autoflow.opt.utils import ConfigurationTransformer
from autoflow.utils.logging_ import get_logger


class BaseDensityEstimator(BaseEstimator):
    def __init__(
            self,
            top_n_percent=15, bandwidth_factor=3, min_bandwidth=1e3,
            bw_estimation="normal_reference", min_points_in_kde=2,
    ):
        self.min_points_in_kde = min_points_in_kde
        self.bw_estimation = bw_estimation
        self.min_bandwidth = min_bandwidth
        self.bandwidth_factor = bandwidth_factor
        self.top_n_percent = top_n_percent
        self.config_transformer: Optional[ConfigurationTransformer]= None
        self.logger=get_logger(self)

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert self.config_transformer is not None
        assert X.shape[1] == len(self.config_transformer.n_choices_list)
        return self

    def predict(self, X: np.ndarray):
        assert self.config_transformer is not None
        assert X.shape[1] == len(self.config_transformer.n_choices_list)

    def process_constants_vector(self, vec: np.ndarray, n_choices, bw, mode="extend"):
        if np.unique(vec).size == 1:
            if n_choices > 1:
                # return vec
                others = set(range(n_choices)) - set(vec)
                other = random.choice(list(others))
            elif n_choices == 0:
                m = vec[0]
                bw = max(0.1, bw)
                while True:
                    other = truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw)
                    other = np.clip(other, 0, 1)
                    if other != m:
                        break
            else:
                raise ValueError
            if mode == "extend":
                return np.hstack([vec, [other]])
            elif mode == "replace":
                vec[0] = other
                return vec
            else:
                raise NotImplementedError
        else:
            return vec

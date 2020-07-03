#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
# from skopt.learning.forest import RandomForestRegressor, ExtraTreesRegressor
# from skopt.learning.gbrt import GradientBoostingQuantileRegressor
import numpy as np
from frozendict import frozendict
from scipy.stats import norm


class EI():
    def __init__(self, xi=0.01):
        self.xi = xi

    def __call__(self, model, X, y_opt):
        mu, std = model.predict(X, return_std=True)
        values = np.zeros_like(mu)
        mask = std > 0
        improve = y_opt - self.xi - mu[mask]
        scaled = improve / std[mask]
        cdf = norm.cdf(scaled)
        pdf = norm.pdf(scaled)
        exploit = improve * cdf
        explore = std[mask] * pdf
        values[mask] = exploit + explore
        return values


class ConfigEvaluator:
    def __init__(self, budget2epm, budget, budget2weight=None, acq_func="EI", acq_func_params=frozendict()):
        self.acq_func_params = dict(acq_func_params)
        if acq_func == "EI":
            acq_func_cls = EI
        else:
            raise NotImplementedError
        self.acq_func = acq_func_cls(**self.acq_func_params)
        self.budget2weight = budget2weight
        self.budget = budget
        self.budget2epm = budget2epm

    def __call__(self, X, y_opt):
        epm = self.budget2epm[self.budget]
        return self.acq_func(epm, X, y_opt)

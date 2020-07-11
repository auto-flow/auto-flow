#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
# from skopt.learning.forest import RandomForestRegressor, ExtraTreesRegressor
# from skopt.learning.gbrt import GradientBoostingQuantileRegressor
import numpy as np
from frozendict import frozendict
from scipy.stats import norm


# todo:  PI LCB

class LogEI():
    def __init__(self, xi=0.01):
        self.xi = xi

    def __call__(self, model, X, y_opt):
        mu, std = model.predict(X, return_std=True)
        var = std ** 2
        values = np.zeros_like(mu)
        mask = std > 0
        f_min = y_opt - self.xi
        improve = f_min - mu[mask]
        # in SMAC, v := scaled
        # smac/optimizer/acquisition.py:388
        scaled = improve / std[mask]
        values[mask] = (np.exp(f_min) * norm.cdf(scaled)) - \
                       (np.exp(0.5 * var[mask] + mu[mask]) * norm.cdf(scaled - std[mask]))
        return values


class EI():
    def __init__(self, xi=0.01):
        # in SMAC, xi=0.0,
        # smac/optimizer/acquisition.py:341
        # par: float=0.0
        # in scikit-optimize, xi=0.01
        # this blog recommend xi=0.01
        # http://krasserm.github.io/2018/03/21/bayesian-optimization/
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
        # You can find the derivation of the EI formula in this blog
        # http://ash-aldujaili.github.io/blog/2018/02/01/ei/
        return values


class ConfigEvaluator:
    def __init__(
            self, budget2epm, budget, budget2weight=None,
            acq_func="EI", acq_func_params=frozendict()
    ):
        self.acq_func_params = dict(acq_func_params)
        # todo: 引入包的形式
        if acq_func == "EI":
            acq_func_cls = EI
        elif acq_func == "LogEI":
            acq_func_cls = LogEI
        else:
            raise NotImplementedError
        self.acq_func = acq_func_cls(**self.acq_func_params)
        self.budget2weight = budget2weight
        self.budget = budget
        self.budget2epm = budget2epm

    def __call__(self, X, y_opt):
        epm = self.budget2epm[self.budget]
        return self.acq_func(epm, X, y_opt)

class DensityConfigEvaluator:
    def __init__(
            self, budget2epm, budget, budget2weight=None
    ):
        self.budget2weight = budget2weight
        self.budget = budget
        self.budget2epm = budget2epm

    def __call__(self, X, y_opt):
        epm = self.budget2epm[self.budget]
        return epm.predict(X)
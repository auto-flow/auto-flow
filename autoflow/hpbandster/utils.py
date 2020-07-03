#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from fractions import Fraction
from typing import Dict, Optional, Union

import numpy as np
from ConfigSpace import ConfigurationSpace, Constant, CategoricalHyperparameter, Configuration
from sklearn.base import BaseEstimator, TransformerMixin

from autoflow.utils.logging_ import get_logger

inc_logger = get_logger("incumbent trajectory")


def get_max_SH_iter(min_budget, max_budget, eta):
    return -int(np.log(min_budget / max_budget) / np.log(eta)) + 1


def modify_timestamps(timestamps: Dict[str, float], delta: float) -> Dict[str, float]:
    result = {}
    for k, v in timestamps.items():
        v += delta
        result[k] = v
    return result


def print_incumbent_trajectory(chal_perf: float, inc_perf: float, challenger: dict, incumbent: dict, budget: float):
    inc_logger.info("Challenger (%.4f) is better than incumbent (%.4f) when budget is (%s)."
                    % (chal_perf, inc_perf, pprint_budget(budget)))
    # Show changes in the configuration
    params = sorted([(param, incumbent.get(param), challenger.get(param))
                     for param in challenger.keys()])
    inc_logger.info("Changes in incumbent:")
    for param in params:
        if param[1] != param[2]:
            inc_logger.info("  %s : %r -> %r" % (param))
        else:
            inc_logger.debug("  %s remains unchanged: %r" %
                             (param[0], param[1]))


def pprint_budget(budget: float):
    if budget - float(int(budget)) == 0:
        return str(int(budget))
    fraction = Fraction.from_float(budget)
    return f"{fraction.numerator}/{fraction.denominator}"


class ConfigSpaceTransformer():
    def __init__(self, impute: Optional[float] = -1, ohe: bool = True):
        self.impute = impute
        self.ohe = ohe

    def fit(self, config_space: ConfigurationSpace):
        mask = []
        n_choices_list = []
        n_constants = 0
        n_variables = 0
        n_top_levels = 0
        for hp in config_space.get_hyperparameters():
            if isinstance(hp, Constant) or (isinstance(hp, CategoricalHyperparameter) and len(hp.choices) == 1):
                # ignore
                mask.append(False)
                n_constants += 1
            else:
                mask.append(True)
                n_variables += 1
                if isinstance(hp, CategoricalHyperparameter):
                    n_choices_list.append(len(hp.choices))
                else:
                    n_choices_list.append(0)
                parents = config_space.get_parents_of(hp.name)
                if len(parents) == 0:
                    n_top_levels += 1

        self.mask = np.array(mask, dtype="bool")
        self.n_choices_list = n_choices_list
        self.n_constants = n_constants
        self.n_variables = n_variables
        self.n_top_levels = n_top_levels
        # todo：判断n_parents
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        vectors = vectors[:, self.mask]
        if self.ohe:
            self.encoder=OneHotEncoder(self.n_choices_list)
            vectors=self.encoder.fit_transform(vectors)

        if self.impute is not None:
            vectors[np.isnan(vectors)] = float(self.impute)
        return vectors

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        assert self.ohe == False
        result = np.zeros([len(self.mask)])
        result[self.mask] = array
        return result


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_choices_list):
        self.n_choices_list = n_choices_list

    def fit(self, X=None, y=None):
        assert len(self.n_choices_list) == X.shape[1]
        return self

    def transform(self, X):
        assert len(self.n_choices_list) == X.shape[1]
        N = X.shape[0]
        result = np.ones(shape=(N, 0), dtype="float32")
        for i, n_choices in enumerate(self.n_choices_list):
            if n_choices == 0:
                col_vector = X[:, i]
            elif n_choices > 0:
                col_vector = np.zeros(shape=(N, n_choices), dtype="float32")
                mask = (~np.isnan(X[:, i]))
                mask_vector = X[:, i][mask]
                col_vector[mask] = np.eye(mask_vector)[mask_vector]
            else:
                raise ValueError
            result = np.hstack((result, col_vector))
        return result

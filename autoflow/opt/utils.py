#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from fractions import Fraction
from typing import Dict, Optional

import numpy as np
from ConfigSpace import ConfigurationSpace, Constant, CategoricalHyperparameter
from sklearn.base import BaseEstimator, TransformerMixin

from autoflow.constants import ERR_LOSS
from autoflow.utils.logging_ import get_logger

inc_logger = get_logger("incumbent trajectory")
logger = get_logger(__name__)


def get_max_SH_iter(min_budget, max_budget, eta):
    return -int(np.log(min_budget / max_budget) / np.log(eta)) + 1


def get_budgets(min_budget, max_budget, eta):
    if min_budget==max_budget:
        return [min_budget]
    budget = min_budget
    budgets = []
    budgets.append(budget)
    # todo: 避免精度造成的影响
    while True:
        budget *= eta
        if budget <= max_budget:
            budgets.append(budget)
            if budget == max_budget:
                break
        else:
            logger.warning(
                f"Invalid budget configuration (min_budget={min_budget},max_budget={max_budget},eta={eta}) , "
                f"final max_budget={budget}, greater than max_budget")
            break
    return budgets


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


class ConfigurationTransformer():
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
            self.encoder = OneHotEncoder(self.n_choices_list)
            vectors = self.encoder.fit_transform(vectors)

        if self.impute is not None:
            if self.impute=="random_choice":
                vectors=self.impute_conditional_data(vectors)
            else:
                vectors[np.isnan(vectors)] = float(self.impute)
        return vectors

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        # todo: 没有考虑涉及OHE的部分
        # fixme: 一般用在对KDE或TPE的l(x)采样后，用vector构建一个Configuration
        assert self.ohe == False
        result = np.zeros([len(self.mask)])
        result[self.mask] = array
        return result

    def impute_conditional_data(self, array):

        return_array = np.empty_like(array)

        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()

            while (np.any(nan_indices)):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()

                if len(valid_indices) > 0:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = np.random.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]

                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.n_choices_list[nan_idx]
                    if t == 0:
                        datum[nan_idx] = np.random.rand()
                    else:
                        datum[nan_idx] = np.random.randint(t)
                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i, :] = datum
        return (return_array)

class LossTransformer(TransformerMixin, BaseEstimator):
    def fit_transform(self, y, *args):
        y = deepcopy(y)
        # cutoff
        y[y >= ERR_LOSS] = y[y < ERR_LOSS].max() + 0.1
        self.y_max = y.max()
        self.y_min = y.min()
        self.y_mean = y.mean()
        self.y_std = y.std()
        self.perc = np.percentile(y, 5)

        return y


class ScaledLossTransformer(LossTransformer):
    def fit_transform(self, y, *args):
        y = super(ScaledLossTransformer, self).fit_transform(y)
        # Subtract the difference between the percentile and the minimum
        y_min = self.y_min - (self.perc - self.y_min)
        # linear scaling
        if self.y_min == self.y_max:
            # prevent diving by zero
            y_min *= 1 - 10 ** -101
        y = (y - y_min) / (self.y_max - self.y_min)
        return y


class LogScaledLossTransformer(LossTransformer):
    def fit_transform(self, y, *args):
        y = super(LogScaledLossTransformer, self).fit_transform(y)
        # Subtract the difference between the percentile and the minimum
        y_min = self.y_min - (self.perc - self.y_min)
        y_min -= 1e-10
        # linear scaling
        if y_min == self.y_max:
            # prevent diving by zero
            y_min *= 1 - (1e-10)
        y = (y - y_min) / (self.y_max - y_min)
        y = np.log(y)
        f_max = y[np.isfinite(y)].max()
        f_min = y[np.isfinite(y)].min()
        y[np.isnan(y)] = f_max
        y[y == -np.inf] = f_min
        y[y == np.inf] = f_max
        return y


if __name__ == '__main__':
    budgets=get_budgets(1/16,1/16,4)
    print(budgets)
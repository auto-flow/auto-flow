#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from fractions import Fraction
from typing import Dict, Optional, Union, List

import numpy as np
from ConfigSpace import ConfigurationSpace, Constant, CategoricalHyperparameter, Configuration
from ConfigSpace.util import deactivate_inactive_hyperparameters
from scipy.spatial.distance import euclidean
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler

from autoflow.constants import ERR_LOSS
from autoflow.utils.logging_ import get_logger

inc_logger = get_logger("incumbent trajectory")
logger = get_logger(__name__)


def is_top_level_activated(config_space, config, hp_name, hp_value=None):
    parent_conditions = config_space.get_parent_conditions_of(hp_name)
    if len(parent_conditions):
        parent_condition = parent_conditions[0]
        parent_value = parent_condition.value
        parent_name = parent_condition.parent.name
        return is_top_level_activated(config_space, config, parent_name, parent_value)
    # 没有条件依赖，就是parent
    if hp_value is None:
        return True
    return config[hp_name] == hp_value


def deactivate(config_space, vector):
    result = deepcopy(vector)
    config = Configuration(config_space, vector=vector)
    for i, hp in enumerate(config_space.get_hyperparameters()):
        name = hp.name
        if not is_top_level_activated(config_space, config, name, None):
            result[i] = np.nan
    result_config = Configuration(configuration_space=config_space, vector=result)
    return result_config


def get_max_SH_iter(min_budget, max_budget, eta):
    return -int(np.log(min_budget / max_budget) / np.log(eta)) + 1


def get_budgets(min_budget, max_budget, eta):
    if min_budget == max_budget:
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
        result = np.ones(shape=(N, 0), dtype="float64")
        for i, n_choices in enumerate(self.n_choices_list):
            if n_choices == 0:
                col_vector = X[:, i][:, None]
            elif n_choices > 0:
                col_vector = np.zeros(shape=(N, n_choices), dtype="float32")
                mask = (~np.isnan(X[:, i]))
                mask_vector = X[:, i][mask]
                col_vector[mask] = np.eye(mask_vector)[mask_vector]
            else:
                raise ValueError
            result = np.hstack((result, col_vector))
        return result


class LabelTsneEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, n_choice: int, n_components="auto", random_state=42):
        self.random_state = random_state
        self.n_choice = n_choice
        assert n_choice > 2, ValueError("'n_choice' must greater than 2.")
        if isinstance(n_components, int):
            assert n_components > 2
            self.n_components = n_components
        elif isinstance(n_components, str):
            self.n_components = max(2, int(round(np.sqrt(n_choice))))
        else:
            raise NotImplementedError
        tsne_matrix = TSNE(random_state=self.random_state, n_components=self.n_components). \
            fit_transform(np.eye(n_choice))
        self.scaler = StandardScaler()
        self.tsne_matrix = self.scaler.fit_transform(tsne_matrix)

    def fit(self, y):
        return self

    def transform(self, y):
        return self.tsne_matrix[y.astype("int32")]

    def inverse_transform(self, X):
        A, M = X.shape
        B, _ = self.tsne_matrix.shape
        assert M == self.n_components
        distance_matrix = np.zeros([A, B])
        Y = self.tsne_matrix
        for i in range(A):
            for j in range(B):
                distance_matrix[i, j] = euclidean(X[i, :], Y[j, :])
        return np.argmin(distance_matrix, axis=1)


class MixedTsneEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, label_tsne_encoders, n_choices_list):
        self.n_choices_list = n_choices_list
        self.label_tsne_encoders = label_tsne_encoders

    def fit(self, X):
        return self

    def transform(self, X):
        N, M = X.shape
        assert M == self.label_tsne_encoders.size
        result = np.ones(shape=(N, 0), dtype="float64")
        for i, label_tsne_encoder in enumerate(self.label_tsne_encoders):
            if label_tsne_encoder:
                col_vector = label_tsne_encoder.transform(X[:, i])
            else:
                col_vector = X[:, i][:, None]
            result = np.hstack((result, col_vector))
        return result

    def inverse_transform(self, X):
        N, M = X.shape
        result = np.ones(shape=(N, 0), dtype="float64")
        start = 0
        for i, (label_tsne_encoder, n_choices) in enumerate(zip(self.label_tsne_encoders, self.n_choices_list)):
            if label_tsne_encoder:
                L = label_tsne_encoder.n_components
                col_vector = label_tsne_encoder.inverse_transform(X[:, start:(start + L)])
                start += L
            elif n_choices == 2:
                col_vector = (X[:, i] > 0.5).astype("float64")
            else:
                col_vector = X[:, i]
            result = np.hstack((result, col_vector[:, None]))
        return result


class ConfigurationTransformer():
    def __init__(self, impute: Optional[float] = -1, ohe: bool = False):
        self.impute = impute
        self.ohe = ohe

    def fit(self, config_space: ConfigurationSpace):
        mask = []
        n_choices_list = []
        n_constants = 0
        n_variables = 0
        n_top_levels = 0
        parents = []
        parent_values = []
        # todo: 划分parents与groups
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
                cur_parents = config_space.get_parents_of(hp.name)
                if len(cur_parents) == 0:
                    n_top_levels += 1
                    parents.append(None)
                    parent_values.append(None)
                else:
                    parents.append(cur_parents[0])
                    parent_conditions = config_space.get_parent_conditions_of(hp.name)
                    parent_condition = parent_conditions[0]
                    parent_values.append(parent_condition.value)
        groups_str = [f"{parent}-{parent_value}" for parent, parent_value in zip(parents, parent_values)]
        group_encoder = LabelEncoder()
        groups = group_encoder.fit_transform(groups_str)
        self.config_space = config_space
        self.groups_str = groups_str
        self.group_encoder = group_encoder
        self.groups = groups
        self.n_groups = np.max(groups) + 1
        self.mask = np.array(mask, dtype="bool")
        self.n_choices_list = n_choices_list
        self.n_constants = n_constants
        self.n_variables = n_variables
        self.n_top_levels = n_top_levels
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        vectors = vectors[:, self.mask]
        if self.ohe:
            self.encoder = OneHotEncoder(self.n_choices_list)
            vectors = self.encoder.fit_transform(vectors)

        if self.impute is not None:
            if self.impute == "random_choice":
                vectors = self.impute_conditional_data(vectors)
            else:
                vectors[np.isnan(vectors)] = float(self.impute)
        return vectors

    def inverse_transform(self, array: np.ndarray, return_vector=False) -> Union[np.ndarray, None, Configuration]:
        # todo: 没有考虑涉及OHE的部分
        # fixme: 一般用在对KDE或TPE的l(x)采样后，用vector构建一个Configuration
        assert self.ohe == False
        result = np.zeros([len(self.mask)])
        result[self.mask] = array
        if return_vector:
            return result
        try:
            config = deactivate(self.config_space, result)
            config = deactivate_inactive_hyperparameters(
                configuration_space=self.config_space,
                configuration=config
            )
            return config
        except Exception as e:
            # print(e)
            # print(config)
            return None

    def impute_conditional_data(self, array):
        # copy from HpBandSter
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


def add_configs_origin(configs: List[Configuration], origin):
    if isinstance(configs, Configuration):
        configs = [configs]
    for config in configs:
        config.origin = origin


def process_config_info_pair(config: Configuration, info_dict: dict):
    info_dict = deepcopy(info_dict)
    if config.origin is None:
        config.origin = "unknown"
    info_dict.update({
        "origin": config.origin
    })
    return config.get_dictionary(), info_dict


if __name__ == '__main__':
    budgets = get_budgets(1 / 16, 1 / 16, 4)
    print(budgets)
    label_tsne_encoder = LabelTsneEncoder(6)
    # label_tsne_encoder.transform(np.array([1, 2, 3, 4, 5]))
    label_tsne_encoder.inverse_transform(np.array([
        [0, 0],
        [1, 1],
    ]))

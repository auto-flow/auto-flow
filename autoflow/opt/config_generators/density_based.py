#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from .bocg import BayesianOptimizationConfigGenerator


class TPE(BayesianOptimizationConfigGenerator):
    epm_cls_name = "TPE"

    def __init__(
            self, config_space, budgets, random_state=None,
            n_samples=5000,
            min_points_in_model=None,
            # TPE parameters
            top_n_percent=15, bandwidth_factor=3, min_bandwidth=1e3,
            bw_estimation="normal_reference",min_points_in_kde=2,
            fill_deactivated_value=False
    ):
        epm_params = dict(
            top_n_percent=top_n_percent,
            bandwidth_factor=bandwidth_factor,
            min_bandwidth=min_bandwidth,
            bw_estimation=bw_estimation,
            min_points_in_kde=min_points_in_kde,
            fill_deactivated_value=fill_deactivated_value
        )
        config_evaluator_params = {
        }
        super(TPE, self).__init__(
            config_space, budgets, random_state,
            self.epm_cls_name, epm_params, "DensityConfigEvaluator", config_evaluator_params,
            min_points_in_model=min_points_in_model,
            config_transformer_params={"impute": None, "ohe": False},
            n_samples=n_samples,
            loss_transformer=None
        )
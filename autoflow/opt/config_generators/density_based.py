#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from .bocg import BayesianOptimizationConfigGenerator


class TPE(BayesianOptimizationConfigGenerator):
    epm_cls_name = "TPE"

    def __init__(
            self, config_space, budgets, random_state=None, initial_points=None,
            n_samples=5000,
            min_points_in_model=None,
            # TPE parameters
            top_n_percent=15, bandwidth_factor=3, min_bandwidth=1e3,
            bw_estimation="normal_reference", min_points_in_kde=2,
            bw_method="scott", cv_times=100, kde_sample_weight_scaler=None,
            fill_deactivated_value=False
    ):
        epm_params = dict(
            top_n_percent=top_n_percent,
            bandwidth_factor=bandwidth_factor,
            min_bandwidth=min_bandwidth,
            bw_estimation=bw_estimation,
            min_points_in_kde=min_points_in_kde,
            fill_deactivated_value=fill_deactivated_value,
            bw_method=bw_method,
            cv_times=cv_times,
            kde_sample_weight_scaler=kde_sample_weight_scaler
        )
        config_evaluator_params = {
        }
        super(TPE, self).__init__(
            config_space, budgets, random_state, initial_points,
            self.epm_cls_name, epm_params, "DensityConfigEvaluator", config_evaluator_params,
            min_points_in_model=min_points_in_model,
            config_transformer_params={"impute": None, "ohe": False},
            n_samples=n_samples,
            loss_transformer=None
        )


class KDE(BayesianOptimizationConfigGenerator):
    epm_cls_name = "KDE"

    def __init__(
            self, config_space, budgets, random_state=None, initial_points=None,
            n_samples=5000,
            min_points_in_model=None,
            # KDE parameters
            top_n_percent=15, bandwidth_factor=3, min_bandwidth=1e3,
            bw_estimation="normal_reference", min_points_in_kde=2
    ):
        epm_params = dict(
            top_n_percent=top_n_percent,
            bandwidth_factor=bandwidth_factor,
            min_bandwidth=min_bandwidth,
            bw_estimation=bw_estimation,
            min_points_in_kde=min_points_in_kde,
        )
        config_evaluator_params = {
        }
        super(KDE, self).__init__(
            config_space, budgets, random_state, initial_points,
            self.epm_cls_name, epm_params, "DensityConfigEvaluator", config_evaluator_params,
            min_points_in_model=min_points_in_model,
            config_transformer_params={"impute": "random_choice", "ohe": False},
            n_samples=n_samples,
            loss_transformer=None
        )

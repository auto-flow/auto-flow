#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from frozendict import frozendict

from .bocg import BayesianOptimizationConfigGenerator


class RF(BayesianOptimizationConfigGenerator):
    epm_cls_name = "RF"

    def __init__(
            self, config_space, budgets, random_state=None,
            acq_func="LogEI", xi=0, kappa=1.96, n_samples=5000,
            min_points_in_model=None,loss_transformer="log_scaled",
            use_local_search=False,
            # TS
            use_thompson_sampling=True, alpha=10, beta=40, top_n_percent=15, hit_top_n_percent=10,
            tpe_params=frozendict(), max_repeated_samples=3, n_candidates=64, sort_by_EI=True,
            # RF parameters
            n_estimators=10, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, min_weight_fraction_leaf=0, max_features="auto",
            max_leaf_nodes=None, min_impurity_decrease=0, bootstrap=True, oob_score=False,
            n_jobs=1, min_variance=0
    ):
        self.loss_transformer = loss_transformer
        epm_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            min_variance=min_variance
        )
        acq_func_params = {}
        if acq_func in ("EI", "LogEI", "PI"):
            acq_func_params.update({"xi": xi})
        elif acq_func == "LCB":
            acq_func_params.update({"kappa": kappa})
        else:
            raise NotImplementedError
        config_evaluator_params = {
            "acq_func": acq_func,
            "acq_func_params": acq_func_params
        }
        super(RF, self).__init__(
            config_space, budgets, random_state,
            self.epm_cls_name, epm_params, "ConfigEvaluator", config_evaluator_params,
            min_points_in_model=min_points_in_model,
            config_transformer_params={"impute": -1, "ohe": False},
            n_samples=n_samples,
            loss_transformer=self.loss_transformer,
            use_local_search=use_local_search,
            use_thompson_sampling=use_thompson_sampling, # for TS
            alpha=alpha,
            beta=beta,
            top_n_percent=top_n_percent,
            hit_top_n_percent=hit_top_n_percent,
            tpe_params=tpe_params,
            max_repeated_samples=max_repeated_samples,
            n_candidates=n_candidates,
            sort_by_EI=sort_by_EI
        )

class ET(RF):
    epm_cls_name = "ET"


class GBRT(BayesianOptimizationConfigGenerator):
    epm_cls_name = "GBRT"

    def __init__(
            self, config_space, budgets, random_state=None,
            acq_func="LogEI", xi=0, kappa=1.96, n_samples=5000,
            min_points_in_model=None,loss_transformer="log_scaled",
            n_jobs=1
    ):
        self.loss_transformer = loss_transformer
        epm_params = dict(
            n_jobs=n_jobs
        )
        acq_func_params = {}
        if acq_func in ("EI", "LogEI", "PI"):
            acq_func_params.update({"xi": xi})
        elif acq_func == "LCB":
            acq_func_params.update({"kappa": kappa})
        else:
            raise NotImplementedError
        config_evaluator_params = {
            "acq_func": acq_func,
            "acq_func_params": acq_func_params
        }
        super(GBRT, self).__init__(
            config_space, budgets, random_state,
            self.epm_cls_name, epm_params, "ConfigEvaluator", config_evaluator_params,
            min_points_in_model=min_points_in_model,
            config_transformer_params={"impute": -1, "ohe": False},
            n_samples=n_samples,
            loss_transformer=self.loss_transformer
        )
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy

import numpy as np
from ConfigSpace import Configuration
from frozendict import frozendict
from skopt.learning.forest import RandomForestRegressor, ExtraTreesRegressor
from skopt.learning.gbrt import GradientBoostingQuantileRegressor

from autoflow.hpbandster.core.config_generator import BaseConfigGenerator
from autoflow.hpbandster.core.dispatcher import Job
from autoflow.hpbandster.optimizers.config_generators.config_evaluator import ConfigEvaluator, DensityConfigEvaluator
from autoflow.hpbandster.optimizers.config_generators.density_estimator.tpe import TreeBasedParzenEstimator
from autoflow.hpbandster.utils import ConfigurationTransformer, LossTransformer, ScaledLossTransformer, \
    LogScaledLossTransformer

epm_str2cls = {
    "ET": ExtraTreesRegressor,
    "RF": RandomForestRegressor,
    "GBRT": GradientBoostingQuantileRegressor,
    "TPE": TreeBasedParzenEstimator,
}


class BayesianOptimizationConfigGenerator(BaseConfigGenerator):
    def __init__(
            self, config_space, budgets, random_state=None, epm="ET", epm_params=frozendict(),
            config_evaluator_cls="ConfigEvaluator", config_evaluator_params=frozendict(),
            min_points_in_model=None, config_transformer_params=None, n_samples=5000,
            loss_transformer=None
    ):
        super(BayesianOptimizationConfigGenerator, self).__init__()
        # ----member variable-----------------------
        if random_state is None:
            random_state = np.random.randint(0, 10000)
        self.random_state = random_state
        self.random_state_seq = random_state
        self.n_samples = n_samples
        self.config_space = config_space
        self.budgets = budgets
        # ----EPM (empirical performance model)------
        self.epm_params = dict(epm_params)
        if isinstance(epm, str):
            epm_cls = epm_str2cls[epm]
        elif isinstance(epm, type):
            epm_cls = epm
        else:
            self.epm = epm
            epm_cls = None
        if epm_cls is not None:
            self.epm = epm_cls(**self.epm_params)
        # ----config_transformer-----------------------
        # todo: 没有必要做这一步，由子类决定
        if config_transformer_params is None:
            if epm in ("ET", "RF", "GBRT"):
                config_transformer_params = {"impute": -1, "ohe": False}
            elif epm in ("GP", "GP-MCMC", "GP-MCMC-NN", "KDE-NN"):
                config_transformer_params = {"impute": 0, "ohe": True}
            elif epm in ("KDE",):
                config_transformer_params = {"impute": "random_choice", "ohe": False}
            elif epm in ("TPE",):
                config_transformer_params = {"impute": None, "ohe": False}
        self.config_transformer_params = dict(config_transformer_params)
        self.config_transformer = ConfigurationTransformer(**self.config_transformer_params)
        self.config_transformer.fit(config_space)
        # set n_choices_list
        if hasattr(self.epm, "n_choices_list"):
            self.epm.n_choices_list = self.config_transformer.n_choices_list
        # ----y_transformer-----------------------
        if loss_transformer is None:
            self.loss_transformer = LossTransformer()
        elif loss_transformer == "log_scaled":
            self.loss_transformer = LogScaledLossTransformer()
        elif loss_transformer == "scaled":
            self.loss_transformer = ScaledLossTransformer()
        else:
            raise NotImplementedError
        # ----min_points_in_model------------------------
        # todo: 这样的判断是否合理？
        if min_points_in_model is None:
            min_points_in_model = max(int(2 * self.config_transformer.n_top_levels), 20)
        self.min_points_in_model = min_points_in_model
        # ----budget to empirical_performance_model, observations, config_evaluator----
        self.budget2epm = {budget: None for budget in budgets}
        self.budget2obvs = {budget: {"losses": [], "configs": [], "vectors": []} for budget in budgets}
        # todo: 更好的形式
        if config_evaluator_cls == "ConfigEvaluator":
            config_evaluator_cls = ConfigEvaluator
        elif config_evaluator_cls=="DensityConfigEvaluator":
            config_evaluator_cls = DensityConfigEvaluator
        else:
            raise NotImplementedError
        self.config_evaluator_cls = config_evaluator_cls
        self.config_evaluator_params = dict(config_evaluator_params)
        self.budget2confevt = {}
        for budget in budgets:
            # todo: budget2weight
            config_evaluator = config_evaluator_cls(self.budget2epm, budget, None, **self.config_evaluator_params)
            self.budget2confevt[budget] = config_evaluator

    def new_result(self, job: Job, update_model=True):
        super().new_result(job)
        ##############################
        ### 1. update observations ###
        ##############################
        if job.result is None:
            # One could skip crashed results, but we decided to
            # assign a +inf loss and count them as bad configurations
            loss = np.inf
        else:
            # same for non numeric losses.
            # Note that this means losses of minus infinity will count as bad!
            loss = job.result["loss"] if np.isfinite(job.result["loss"]) else np.inf
        budget = job.kwargs["budget"]
        config_dict = job.kwargs["config"]
        config = Configuration(self.config_space, config_dict)
        self.budget2obvs[budget]["configs"].append(deepcopy(config))
        self.budget2obvs[budget]["vectors"].append(config.get_array())
        self.budget2obvs[budget]["losses"].append(loss)
        losses = self.budget2obvs[budget]["losses"]
        vectors = np.array(self.budget2obvs[budget]["vectors"])
        # If the number of observations is too few, the condition of training model is not satisfied
        if len(losses) < self.min_points_in_model:
            return
        if not update_model:
            return
        ##############################################################################
        ### 2. transform X_obvs, do one-hot-encoding, imputing or other operators  ###
        ##############################################################################
        X_obvs = self.config_transformer.transform(np.array(vectors))
        y_obvs = self.loss_transformer.fit_transform(np.array(losses))

        ################################################
        ### 3. training empirical performance model  ###
        ################################################
        if self.budget2epm[budget] is None:
            self.budget2epm[budget] = deepcopy(self.epm)
        self.budget2epm[budget].fit(X_obvs, y_obvs)

    def get_config(self, budget):
        budget = self.get_available_max_budget()
        epm = self.budget2epm[budget]
        if epm is None:
            max_sample = 1000
            i = 0
            info_dict = {"model_based_pick": False}
            while i < max_sample:
                i += 1
                config = self.config_space.sample_configuration()
                if self.is_config_exist(budget, config):
                    self.logger.info(
                        f"The sample already exists and needs to be resampled. It's the {i}-th time sampling in random search. ")
                else:
                    return config.get_dictionary(), info_dict
            # todo: 收纳这个代码块
            seed = np.random.randint(1, 8888)
            self.config_space.seed(seed)
            config = self.config_space.sample_configuration()
            info_dict.update({"sampling_different_samples_failed": True, "seed": seed})
            return config.get_dictionary(), info_dict

        info_dict = {"model_based_pick": True}
        config_evaluator = self.budget2confevt[budget]
        # 让config_evaluator给所有的随机样本打分
        configs = self.config_space.sample_configuration(self.n_samples)
        # todo: 用 g-means 增加随机性
        X = np.array([config.get_array() for config in configs], dtype="float32")
        X_trans = self.config_transformer.transform(X)
        y_opt = np.min(self.budget2obvs[budget]["losses"])
        rewards = config_evaluator(X_trans, y_opt)
        np.random.seed(self.random_state_seq)
        self.random_state_seq += 1
        random = np.random.rand(len(rewards))
        indexes = np.lexsort((random.flatten(), -rewards.flatten()))
        # 选取获益最大，且没有出现过的一个配置
        # todo: 局部搜索
        for i, index in enumerate(indexes):
            config = configs[index]
            if self.is_config_exist(budget, configs[index]):
                self.logger.info(
                    f"The sample already exists and needs to be resampled. It's the {i}-th time sampling in bayesian search. ")
            else:
                return config.get_dictionary(), info_dict
        # todo: 收纳这个代码块
        seed = np.random.randint(1, 8888)
        self.config_space.seed(seed)
        config = self.config_space.sample_configuration()
        info_dict.update({"sampling_different_samples_failed": True, "seed": seed})
        return config.get_dictionary(), info_dict

    def is_config_exist(self, budget, config: Configuration):
        vectors = np.array(self.budget2obvs[budget]["vectors"])
        if len(vectors) == 0:
            return False
        vectors[np.isnan(vectors)] = -1
        vector = deepcopy(config.get_array())
        vector[np.isnan(vector)] = -1
        if np.any(np.all(vector == vectors, axis=1)):
            return True
        return False

    def get_available_max_budget(self):
        sorted_budgets = sorted(self.budget2epm.keys())
        for budget in sorted(self.budget2epm.keys()):
            if self.budget2epm[budget] is not None:
                return budget
        return sorted_budgets[0]

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
from autoflow.hpbandster.optimizers.config_generators.config_evaluator import ConfigEvaluator
from autoflow.hpbandster.utils import ConfigSpaceTransformer


class BayesianOptimizationConfigGenerator(BaseConfigGenerator):
    def __init__(
            self, config_space, budgets, random_state=None,epm="ET", epm_params=frozendict(),
            config_evaluator_cls="ConfigEvaluator", config_evaluator_params=frozendict(),
            min_points_in_model=None, config_transformer_params=None,n_samples=5000
    ):
        super(BayesianOptimizationConfigGenerator, self).__init__()
        # ----member variable-----------------------
        if random_state is None:
            random_state=np.random.randint(0,10000)
        self.random_state = random_state
        self.random_state_seq = random_state
        self.n_samples = n_samples
        self.config_space = config_space
        self.budgets = budgets
        # ----EPM (empirical performance model)------
        self.epm_params = dict(epm_params)
        if epm == "ET":
            epm_cls = ExtraTreesRegressor
            config_transformer_params = {"impute": -1, "ohe": False}
        elif epm == "RF":
            epm_cls = RandomForestRegressor
        elif epm == "GBRT":
            epm_cls = GradientBoostingQuantileRegressor
        else:
            raise NotImplementedError
        self.epm = epm_cls(**self.epm_params)
        # ----config_transformer-----------------------
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
        self.config_transformer = ConfigSpaceTransformer(**self.config_transformer_params)
        self.config_transformer.fit(config_space)
        # ----min_points_in_model------------------------
        if min_points_in_model is None:
            min_points_in_model = min(int(2 * self.config_transformer.n_top_levels), 20)
        self.min_points_in_model = min_points_in_model
        # ----budget to empirical_performance_model, observations, config_evaluator----
        self.budget2epm = dict(zip(budgets, [deepcopy(self.epm)] * len(budgets)))
        self.budget2obvs = dict(zip(budgets, [{"losses": [], "configs": [], "vectors": []}] * len(budgets)))
        if config_evaluator_cls == "ConfigEvaluator":
            config_evaluator_cls = ConfigEvaluator
        else:
            raise NotImplementedError
        self.config_evaluator_cls = config_evaluator_cls
        self.config_evaluator_params = dict(config_evaluator_params)
        self.budget2confevt = {}
        for budget in budgets:
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
        config = Configuration(self.config_space, job.kwargs["config"])
        self.budget2obvs[budget]["configs"].append(config)
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
        y_obvs = np.array(losses)

        ################################################
        ### 3. training empirical performance model  ###
        ################################################
        self.budget2epm[budget].fit(X_obvs, y_obvs)

    def get_config(self, budget):

        epm=self.budget2epm[budget]
        if epm is None:
            max_sample = 1000
            i = 0
            info_dict={"model_based_pick":False}
            while i < max_sample :
                i += 1
                config= self.config_space.sample_configuration()
                if self.is_config_exist(budget,config):
                    self.logger.info(
                        f"The sample already exists and needs to be resampled. It's the {i}-th time sampling in random search. ")
                else:
                    return config.get_dictionary(),info_dict
            seed = np.random.randint(1, 8888)
            self.config_space.seed(seed)
            config=self.config_space.sample_configuration()
            info_dict.update({"sampling_different_samples_failed":True,"seed":seed})
            return config.get_dictionary(),info_dict

        info_dict = {"model_based_pick": True}
        config_evaluator=self.budget2confevt[budget]
        # 让config_evaluator给所有的随机样本打分
        configs=self.config_space.sample_configuration(self.n_samples)
        X=np.array([config.get_array() for config in configs], dtype="float32")
        X_trans=self.config_transformer.transform(X)
        y_opt=np.min(self.budget2obvs[budget]["losses"])
        rewards=config_evaluator(X_trans,y_opt)
        np.random.seed(self.random_state)
        random = np.random.rand(len(rewards))
        indexes = np.lexsort((random.flatten(), -rewards.flatten()))
        # 选取获益最大，且没有出现过的一个配置
        # todo: 局部搜索
        for i,index in enumerate(indexes):
            config=configs[index]
            if self.is_config_exist(budget,configs[index]):
                self.logger.info(
                    f"The sample already exists and needs to be resampled. It's the {i}-th time sampling in bayesian search. ")
            else:
                return config.get_dictionary(),info_dict
        seed = np.random.randint(1, 8888)
        self.config_space.seed(seed)
        config = self.config_space.sample_configuration()
        info_dict.update({"sampling_different_samples_failed": True, "seed": seed})
        return config.get_dictionary(), info_dict


    def is_config_exist(self,budget,config:Configuration):
        vectors=np.ndarray(self.budget2obvs[budget]["vectors"])
        if len(vectors)==0:
            return False
        vectors[np.isnan(vectors)]=-1
        vector=config.get_array()
        vector[np.isnan(vectors)]=-1
        if np.any(np.all(vector == vectors, axis=1)):
            return True
        return False





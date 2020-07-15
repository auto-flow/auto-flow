#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import itertools
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List

import numpy as np
from ConfigSpace import Configuration
from ConfigSpace.util import get_one_exchange_neighbourhood
from frozendict import frozendict
from scipy.stats import beta
from skopt.learning.forest import RandomForestRegressor, ExtraTreesRegressor
from skopt.learning.gbrt import GradientBoostingQuantileRegressor

get_one_exchange_neighbourhood = partial(get_one_exchange_neighbourhood, stdev=0.05, num_neighbors=8)
from autoflow.utils.klass import get_valid_params_in_kwargs
from .base import BaseConfigGenerator
from .config_evaluator import ConfigEvaluator, DensityConfigEvaluator
from .density_estimator.kde import KDE4BO
from .density_estimator.tpe import TreeStructuredParzenEstimator
from ..structure import Job
from ..utils import ConfigurationTransformer, LossTransformer, ScaledLossTransformer, \
    LogScaledLossTransformer

epm_str2cls = {
    "ET": ExtraTreesRegressor,
    "RF": RandomForestRegressor,
    "GBRT": GradientBoostingQuantileRegressor,
    "TPE": TreeStructuredParzenEstimator,
    "KDE": KDE4BO
}


class BayesianOptimizationConfigGenerator(BaseConfigGenerator):
    def __init__(
            self, config_space, budgets, random_state=None, epm="ET", epm_params=frozendict(),
            config_evaluator_cls="ConfigEvaluator", config_evaluator_params=frozendict(),
            min_points_in_model=None, config_transformer_params=None, n_samples=5000,
            loss_transformer=None,
            use_local_search=False,
            use_thompson_sampling=True, alpha=10, beta=40, top_n_percent=15, hit_top_n_percent=8,
            tpe_params=frozendict(), max_repeated_samples=3, n_candidates=64, sort_by_EI=True
    ):
        super(BayesianOptimizationConfigGenerator, self).__init__()
        # ----member variable-----------------------
        self.use_local_search = use_local_search
        self.sort_by_EI = sort_by_EI
        self.n_candidates = n_candidates
        self.max_repeated_samples = max_repeated_samples
        self.tpe_params = dict(tpe_params)
        self.hit_top_n_percent = hit_top_n_percent
        self.top_n_percent = top_n_percent
        self.beta = beta
        self.alpha = alpha
        self.use_thompson_sampling = use_thompson_sampling
        if random_state is None:
            random_state = np.random.randint(0, 10000)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
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
        if hasattr(self.epm, "config_transformer"):
            self.epm.set_config_transformer(self.config_transformer)
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
        # ----TPE------------------------
        self.tpe_params.update({"top_n_percent": self.top_n_percent})
        self.tpe = TreeStructuredParzenEstimator(**self.tpe_params)
        self.tpe_config_transformer = ConfigurationTransformer(impute=False, ohe=False)
        self.tpe_loss_transformer = LossTransformer()
        self.tpe_config_transformer.fit(self.config_space)
        self.tpe.set_config_transformer(self.tpe_config_transformer)
        # ----budget to empirical_performance_model, observations, config_evaluator----
        self.budget2epm = {budget: None for budget in budgets}
        self.budget2obvs = {budget: {"losses": [], "configs": [], "vectors": []} for budget in budgets}
        # todo: 更好的形式
        if config_evaluator_cls == "ConfigEvaluator":
            config_evaluator_cls = ConfigEvaluator
        elif config_evaluator_cls == "DensityConfigEvaluator":
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
        config_info = job.kwargs["config_info"]
        config = Configuration(self.config_space, config_dict)

        ###############################
        ### 3. update observations  ###
        ###############################
        self.budget2obvs[budget]["configs"].append(deepcopy(config))
        self.budget2obvs[budget]["vectors"].append(config.get_array())
        self.budget2obvs[budget]["losses"].append(loss)
        losses = np.array(self.budget2obvs[budget]["losses"])
        vectors = np.array(self.budget2obvs[budget]["vectors"])
        #####################################
        ### 2. update beta distributions  ###
        #####################################
        if config_info.get("thompson_sampling", False):
            sorted_losses = np.sort(losses)
            L = losses.size
            if loss <= sorted_losses[max(1, int(round(L * (self.hit_top_n_percent / 100))))]:
                state = "hit"
                self.alpha += 1
            else:
                state = "miss"
                self.beta += 1
            self.logger.info(f"Updated beta distributions, state = {state}, alpha = {self.alpha}, beta = {self.beta} .")
        ###################################################################
        ### 3. Judge whether the EPM training conditions are satisfied  ###
        ###################################################################
        # If the number of observations is too few, the condition of training model is not satisfied
        if len(losses) < self.min_points_in_model:
            return
        if not update_model:
            return
        ##############################################################################
        ### 4. transform X_obvs, do one-hot-encoding, imputing or other operators  ###
        ##############################################################################
        X_obvs = self.config_transformer.transform(vectors)
        y_obvs = self.loss_transformer.fit_transform(losses)
        ################################################
        ### 5. training empirical performance model  ###
        ################################################
        if self.budget2epm[budget] is None:
            epm = deepcopy(self.epm)
        else:
            epm = self.budget2epm[budget]
        self.budget2epm[budget] = epm.fit(X_obvs, y_obvs)

    def transform(self, configs: List[Configuration]):
        X = np.array([config.get_array() for config in configs], dtype="float32")
        X_trans = self.config_transformer.transform(X)
        return X_trans

    def evaluate(self, configs: List[Configuration], budget, y_opt=None,
                 return_loss_config_pairs=False, return_loss=False, return_loss_config=False):
        config_evaluator = self.budget2confevt[budget]
        if isinstance(configs, Configuration):
            configs = [configs]
        X_trans = self.transform(configs)
        if y_opt is None:
            y_opt = self.get_y_opt(budget)
        rewards = config_evaluator(X_trans, y_opt)
        random_var = self.rng.rand(len(rewards))
        indexes = np.lexsort((random_var.flatten(), -rewards.flatten()))
        rewards_sorted = rewards[indexes]
        configs_sorted = [configs[ix] for ix in indexes]
        if return_loss_config_pairs:
            return list(zip(-rewards_sorted, configs_sorted))
        if return_loss:
            return -rewards
        if return_loss_config:
            return -rewards_sorted, configs_sorted
        return configs_sorted

    def get_y_opt(self, budget):
        y_opt = np.min(self.budget2obvs[budget]["losses"])
        return y_opt

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
                    self.logger.info(f"The sample already exists and needs to be resampled. "
                                     f"It's the {i}-th time sampling in random sampling. ")
                else:
                    return config.get_dictionary(), info_dict
            # todo: 收纳这个代码块
            seed = self.rng.randint(1, 8888)
            self.config_space.seed()
            config = self.config_space.sample_configuration()
            info_dict.update({"sampling_different_samples_failed": True, "seed": seed})
            return config.get_dictionary(), info_dict

        info_dict = {"model_based_pick": True}
        # thompson sampling
        if self.use_thompson_sampling:
            ts_config, ts_info_dict = self.thompson_sampling(budget, info_dict)
            if ts_config is not None:
                self.logger.info("Using thompson sampling near the dominant samples.")
                return ts_config.get_dictionary(), ts_info_dict
        # 让config_evaluator给所有的随机样本打分
        configs = self.config_space.sample_configuration(self.n_samples)
        losses, configs_sorted = self.evaluate(configs, budget, return_loss_config=True)
        if self.use_local_search:
            start_points = self.get_local_search_initial_points(budget, 10, configs_sorted)  # todo 最后把以前跑过的样本删掉
            local_losses, local_configs = self.local_search(start_points,
                                                            budget)  # todo: 判断start_points 与local_configs的关系
            concat_losses = np.hstack([losses.flatten(), local_losses.flatten()])
            concat_configs = configs + local_configs
            random_var = self.rng.rand(len(concat_losses))
            indexes = np.lexsort((random_var.flatten(), concat_losses))
            concat_configs_sorted = [concat_configs[i] for i in indexes]
            concat_losses = concat_losses[indexes]
        else:
            concat_losses, concat_configs_sorted = losses, configs_sorted
        # 选取获益最大，且没有出现过的一个配置
        # todo: 局部搜索
        for i, config in enumerate(concat_configs_sorted):
            if self.is_config_exist(budget, config):
                self.logger.info(f"The sample already exists and needs to be resampled. "
                                 f"It's the {i}-th time sampling in bayesian sampling. ")
                # 超过 max_repeated_samples ， 用TS算法采样
                if i >= self.max_repeated_samples and self.use_thompson_sampling:
                    ts_config, ts_info_dict = self.thompson_sampling(budget, info_dict, True)
                    return ts_config.get_dictionary(), ts_info_dict
            else:
                return config.get_dictionary(), info_dict
        # todo: 收纳这个代码块
        seed = self.rng.randint(1, 8888)
        self.config_space.seed(seed)
        config = self.config_space.sample_configuration()
        info_dict.update({"sampling_different_samples_failed": True, "seed": seed})
        return config.get_dictionary(), info_dict

    def get_local_search_initial_points(self, budget, num_points, additional_start_points):
        # 对之前的样本做评价
        # 1. 按acq排序，前num_points的历史样本
        config_evaluator = self.budget2confevt[budget]
        configs_previous_runs = self.budget2obvs[budget]["configs"]
        X_trans = self.transform(configs_previous_runs)
        y_opt = np.min(self.budget2obvs[budget]["losses"])
        rewards = config_evaluator(X_trans, y_opt)
        # 只取前num_points的样本
        random_var = self.rng.rand(len(rewards))
        indexes = np.lexsort((random_var.flatten(), -rewards.flatten()))
        configs_previous_runs_sorted_by_acq = [configs_previous_runs[ix] for ix in indexes[:num_points]]
        # 2. 按loss排序，前num_points的历史样本
        losses = np.array(self.budget2obvs[budget]["losses"])
        random_var = self.rng.rand(len(losses))
        indexes = np.lexsort((random_var.flatten(), losses.flatten()))
        configs_previous_runs_sorted_by_loss = [configs_previous_runs[ix] for ix in indexes[:num_points]]
        additional_start_points = additional_start_points[:num_points]
        init_points = []
        init_points_as_set = set()
        for cand in itertools.chain(
                configs_previous_runs_sorted_by_acq,
                configs_previous_runs_sorted_by_loss,
                additional_start_points,
        ):
            if cand not in init_points_as_set:
                init_points.append(cand)
                init_points_as_set.add(cand)

        return init_points

    def local_search(
            self,
            start_points: List[Configuration],
            budget,
    ) -> Tuple[np.ndarray, List[Configuration]]:
        y_opt = self.get_y_opt(budget)
        # Compute the acquisition value of the incumbents
        num_incumbents = len(start_points)
        acq_val_incumbents_, incumbents = self.evaluate(deepcopy(start_points), budget, y_opt,
                                                        return_loss_config=True)
        acq_val_incumbents: list = acq_val_incumbents_.tolist()

        # Set up additional variables required to do vectorized local search:
        # whether the i-th local search is still running
        active = [True] * num_incumbents
        # number of plateau walks of the i-th local search. Reaching the maximum number is the stopping criterion of
        # the local search.
        n_no_plateau_walk = [0] * num_incumbents
        # tracking the number of steps for logging purposes
        local_search_steps = [0] * num_incumbents
        # tracking the number of neighbors looked at for logging purposes
        neighbors_looked_at = [0] * num_incumbents
        # tracking the number of neighbors generated for logging purposse
        neighbors_generated = [0] * num_incumbents
        # how many neighbors were obtained for the i-th local search. Important to map the individual acquisition
        # function values to the correct local search run
        # todo
        self.vectorization_min_obtain = 2
        self.n_steps_plateau_walk = 10
        self.vectorization_max_obtain = 64
        # todo
        obtain_n = [self.vectorization_min_obtain] * num_incumbents
        # Tracking the time it takes to compute the acquisition function
        times = []

        # Set up the neighborhood generators
        neighborhood_iterators = []
        for i, inc in enumerate(incumbents):
            neighborhood_iterators.append(get_one_exchange_neighbourhood(
                inc, seed=self.rng.randint(low=0, high=100000)))
            local_search_steps[i] += 1
        # Keeping track of configurations with equal acquisition value for plateau walking
        neighbors_w_equal_acq = [[]] * num_incumbents

        num_iters = 0
        while np.any(active):

            num_iters += 1
            # Whether the i-th local search improved. When a new neighborhood is generated, this is used to determine
            # whether a step was made (improvement) or not (iterator exhausted)
            improved = [False] * num_incumbents
            # Used to request a new neighborhood for the incumbent of the i-th local search
            new_neighborhood = [False] * num_incumbents

            # gather all neighbors
            neighbors = []
            for i, neighborhood_iterator in enumerate(neighborhood_iterators):
                if active[i]:
                    neighbors_for_i = []
                    for j in range(obtain_n[i]):
                        try:
                            n = next(neighborhood_iterator)  # n : Configuration
                            neighbors_generated[i] += 1
                            neighbors_for_i.append(n)
                        except StopIteration:
                            obtain_n[i] = len(neighbors_for_i)
                            new_neighborhood[i] = True
                            break
                    neighbors.extend(neighbors_for_i)

            if len(neighbors) != 0:
                acq_val = self.evaluate(neighbors, budget, return_loss=True)
                if np.ndim(acq_val.shape) == 0:
                    acq_val = [acq_val]

                # Comparing the acquisition function of the neighbors with the acquisition value of the incumbent
                acq_index = 0
                # Iterating the all i local searches
                for i in range(num_incumbents):
                    if not active[i]:
                        continue
                    # And for each local search we know how many neighbors we obtained
                    for j in range(obtain_n[i]):
                        # The next line is only true if there was an improvement and we basically need to iterate to
                        # the i+1-th local search
                        if improved[i]:
                            acq_index += 1
                        else:
                            neighbors_looked_at[i] += 1

                            # Found a better configuration
                            if acq_val[acq_index] < acq_val_incumbents[i]:
                                self.logger.debug(
                                    "Local search %d: Switch to one of the neighbors (after %d configurations).",
                                    i,
                                    neighbors_looked_at[i],
                                )
                                incumbents[i] = neighbors[acq_index]
                                acq_val_incumbents[i] = acq_val[acq_index]
                                new_neighborhood[i] = True
                                improved[i] = True
                                local_search_steps[i] += 1
                                neighbors_w_equal_acq[i] = []
                                obtain_n[i] = 1
                            # Found an equally well performing configuration, keeping it for plateau walking
                            elif acq_val[acq_index] == acq_val_incumbents[i]:
                                neighbors_w_equal_acq[i].append(neighbors[acq_index])

                            acq_index += 1

            # Now we check whether we need to create new neighborhoods and whether we need to increase the number of
            # plateau walks for one of the local searches. Also disables local searches if the number of plateau walks
            # is reached (and all being switched off is the termination criterion).
            for i in range(num_incumbents):
                if not active[i]:
                    continue
                if obtain_n[i] == 0 or improved[i]:
                    obtain_n[i] = 2
                else:
                    obtain_n[i] = obtain_n[i] * 2
                    obtain_n[i] = min(obtain_n[i], self.vectorization_max_obtain)
                if new_neighborhood[i]:
                    if not improved[i] and n_no_plateau_walk[i] < self.n_steps_plateau_walk:
                        if len(neighbors_w_equal_acq[i]) != 0:
                            incumbents[i] = neighbors_w_equal_acq[i][0]
                            neighbors_w_equal_acq[i] = []
                        n_no_plateau_walk[i] += 1
                    if n_no_plateau_walk[i] >= self.n_steps_plateau_walk:
                        active[i] = False
                        continue

                    neighborhood_iterators[i] = get_one_exchange_neighbourhood(
                        incumbents[i], seed=self.rng.randint(low=0, high=100000),
                    )

        self.logger.debug(
            "Local searches took %s steps and looked at %s configurations. Computing the acquisition function in "
            "vectorized for took %f seconds on average.",
            local_search_steps, neighbors_looked_at, np.mean(times),
        )
        # todo: origin 标注来自局部搜索
        return np.array(acq_val_incumbents), incumbents
        # return [(a, i) for a, i in zip(acq_val_incumbents, incumbents)]

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

    def thompson_sampling(self, budget, info_dict, hit=False) -> Tuple[Optional[Configuration], Optional[dict]]:
        # todo: 增加 n_candidates, sort_by_EI参数
        info_dict = deepcopy(info_dict)
        rv = beta(self.alpha, self.beta)
        prob = rv.rvs()
        # Samples were taken near the dominant samples
        if (self.rng.rand() < prob) or hit:
            epm = self.budget2epm[budget]
            if hasattr(epm, "sample"):
                sampler = epm
            else:
                vectors = np.array(self.budget2obvs[budget]["vectors"])
                X_obvs = self.tpe_config_transformer.transform(vectors)
                losses = np.array(self.budget2obvs[budget]["losses"])
                y_obvs = self.tpe_loss_transformer.fit_transform(losses)
                self.tpe.fit(X_obvs, y_obvs)
                sampler = self.tpe
            kwargs = {"n_candidates": self.n_candidates, "sort_by_EI": self.sort_by_EI}
            samples = sampler.sample(**get_valid_params_in_kwargs(sampler.sample, kwargs))
            for i, sample in enumerate(samples):
                if self.is_config_exist(budget, sample):
                    self.logger.info(f"The sample already exists and needs to be resampled. "
                                     f"It's the {i}-th time sampling in thompson sampling. ")
                else:
                    info_dict.update({"thompson_sampling": True})
                    return sample, info_dict
            sample = self.config_space.sample_configuration()
            info_dict = {"model_based_pick": False}
            return sample, info_dict
        return None, None

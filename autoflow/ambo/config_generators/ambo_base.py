#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import itertools
import json
import math
import os
from collections import defaultdict
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pylab as plt
from ConfigSpace import Configuration
from ConfigSpace.util import get_one_exchange_neighbourhood
from frozendict import frozendict
from scipy.stats import beta
from sklearn.manifold import TSNE
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils._testing import ignore_warnings
from skopt.learning.forest import RandomForestRegressor, ExtraTreesRegressor
from skopt.learning.gbrt import GradientBoostingQuantileRegressor

from autoflow.feature_engineer.encode.equidistance import EquidistanceEncoder
from autoflow.utils.logging_ import get_logger
from generic_fs.local import LocalFS

get_one_exchange_neighbourhood = partial(get_one_exchange_neighbourhood, stdev=0.05, num_neighbors=8)
from autoflow.utils.klass import get_valid_params_in_kwargs
from .config_evaluator import ConfigEvaluator, DensityConfigEvaluator
from .density_estimator.kde import KDE4BO
from .density_estimator.tpe import TreeStructuredParzenEstimator
from ..structure import Job
from ..utils import ConfigurationTransformer, LossTransformer, ScaledLossTransformer, \
    LogScaledLossTransformer, add_configs_origin

epm_str2cls = {
    "ET": ExtraTreesRegressor,
    "RF": RandomForestRegressor,
    "GBRT": GradientBoostingQuantileRegressor,
    "TPE": TreeStructuredParzenEstimator,
    "KDE": KDE4BO
}


class BaseConfigGenerator(object):
    """
    The config generator determines how new configurations are sampled. This can take very different levels of
    complexity, from random sampling to the construction of complex empirical prediction models for promising
    configurations.
    """

    def __init__(self):
        """
        Parameters
        ----------

        directory: string
            where the results are logged
        logger: ambo.utils.result_logger_v??
            the logger to store the data, defaults to v1
        overwrite: bool
            whether or not existing data will be overwritten
        logger: logging.logger
            for some debug output

        """

        self.logger = get_logger(self)

    def get_config(self, budget):
        """
        function to sample a new configuration

        This function is called inside Hyperband to query a new configuration

        Parameters
        ----------
        budget: float
            the budget for which this configuration is scheduled

        returns: (config, info_dict)
            must return a valid configuration and a (possibly empty) info dict
        """
        raise NotImplementedError('This function needs to be overwritten in %s.' % (self.__class__.__name__))

    def new_result(self, job, update_model=True):
        """
        registers finished runs

        Every time a run has finished, this function should be called
        to register it with the result logger. If overwritten, make
        sure to call this method from the base class to ensure proper
        logging.


        Parameters
        ----------
        job: instance of ambo.distributed.dispatcher.Job
            contains all necessary information about the job
        update_model: boolean
            determines whether a model inside the config_generator should be updated
        """
        if not job.exception is None:
            self.logger.warning("job {} failed with exception\n{}".format(job.id, job.exception))


class AdaptiveMetaBayesianOptimization(BaseConfigGenerator):
    def __init__(
            self, config_space, budgets, random_state=None, initial_points=None, fs=None, record_path=None,
            epm="ET", epm_params=frozendict(),
            config_evaluator_cls="ConfigEvaluator", config_evaluator_params=frozendict(),
            min_points_in_model=15, config_transformer_params=None, n_samples=5000,
            loss_transformer=None,
            use_local_search=False,
            # TPE thompson sampling
            use_thompson_sampling=1, alpha=10, beta=40, top_n_percent=15, hit_top_n_percent=10,
            tpe_params=frozendict(), max_repeated_samples=3, n_candidates=64, sort_by_EI=True,
            meta_encoder=None, bandwidth_factor=3,
            # meta learn and multi task learn
            support_ensemble_epms=True, update_weight_accepted_samples=10, update_weight_steps=3,
            ensemble_test_size=0.33,
    ):
        super(AdaptiveMetaBayesianOptimization, self).__init__()
        # ----member variable-----------------------
        self.bandwidth_factor = bandwidth_factor
        self.update_weight_steps = update_weight_steps
        self.ensemble_test_size = ensemble_test_size
        self.meta_encoder = meta_encoder
        self.initial_points = initial_points
        self.initial_points_index = 0
        self.update_weight_accepted_samples = max(5, update_weight_accepted_samples)
        # if len(budgets) <= 1 and support_ensemble_epms:
        #     self.logger.warning(
        #         f"'support_ensemble_epms' is True but len(budgets) <= 1 ,  This value is corrected to False.")
        #     support_ensemble_epms = False
        self.support_ensemble_epms = support_ensemble_epms
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
        self.has_entity_encoder_ = None
        # ----File System----------------------------
        if fs is None:
            fs = LocalFS()
        self.fs = fs
        if record_path is None:
            record_path = "~/autoflow/ambo"
        self.record_path = os.path.expandvars(os.path.expanduser(record_path))
        self.fs.mkdir(self.record_path)
        self.entity_path = fs.join(self.record_path, "entity_encoder")
        self.fs.mkdir(self.entity_path)
        self.weight_path = fs.join(self.record_path, "epms_weight")
        self.fs.mkdir(self.weight_path)
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
        self.config_transformer_params = dict(config_transformer_params)
        self.config_transformer = ConfigurationTransformer(**self.config_transformer_params)
        self.config_transformer.fit(config_space)
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
        # if min_points_in_model is None:
        #     min_points_in_model = max(int(2 * self.config_transformer.n_top_levels), 20)
        # if self.support_ensemble_epms:
        #     min_points_in_model += self.update_weight_steps
        self.min_points_in_model = min_points_in_model
        # ----TPE------------------------
        self.tpe_params.update({"top_n_percent": self.top_n_percent})
        self.tpe = TreeStructuredParzenEstimator(**self.tpe_params)
        if self.meta_encoder is None:
            encoder = EquidistanceEncoder()
        else:
            encoder = self.meta_encoder
        self.tpe_config_transformer = ConfigurationTransformer(impute=None, encoder=encoder)
        self.tpe_loss_transformer = LossTransformer()
        self.tpe_config_transformer.fit(self.config_space)
        if self.meta_encoder is None:
            vectors=np.array([config.get_array() for config in self.config_space.sample_configuration(5000)])
            self.tpe_config_transformer.fit_encoder(vectors)
        self.tpe.set_config_transformer(self.tpe_config_transformer)
        # ----budget to empirical_performance_model, observations, config_evaluator----
        self.budget2epm = {budget: None for budget in budgets}
        self.budget2obvs = {budget: {"losses": [], "configs": [], "vectors": [], "locks": []} for budget in budgets}
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
            config_evaluator = config_evaluator_cls(self.budget2epm, budget, **self.config_evaluator_params)
            self.budget2confevt[budget] = config_evaluator
        self.update_weight_cnt = 0

    def new_result(self, job: Job, update_model=True, should_update_weight=0):
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
        # add lock (It may be added twice, but it does not affect)
        self.budget2obvs[budget]["locks"].append(config.get_array().copy())
        self.budget2obvs[budget]["configs"].append(deepcopy(config))
        self.budget2obvs[budget]["vectors"].append(config.get_array())
        self.budget2obvs[budget]["losses"].append(loss)
        losses = np.array(self.budget2obvs[budget]["losses"])
        vectors = np.array(self.budget2obvs[budget]["vectors"])
        ###################################################################
        ### 2. Judge whether the EPM training conditions are satisfied  ###
        ###################################################################
        # If the number of observations is too few, the condition of training model is not satisfied
        if len(losses) < self.min_points_in_model:
            return
        if not update_model:
            return
        #####################################
        ### 3. update beta distributions  ###
        #####################################
        if config_info.get("thompson_sampling", False):
            sorted_losses = np.sort(losses)
            L = losses.size
            if L >= 5:
                if loss <= sorted_losses[max(0, int(round(L * (self.hit_top_n_percent / 100))))]:
                    state = "hit"
                    self.alpha += 1
                else:
                    state = "miss"
                    self.beta += 1
                self.logger.info(f"Updated beta distributions, state = {state}, "
                                 f"alpha = {self.alpha}, beta = {self.beta} .")
        ######################################################################
        ### 4. transform X_obvs, do encoding, imputing or other operators  ###
        ######################################################################
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
        ###########################################
        ### 6. update surragete model's weight  ###
        ###########################################
        self.update_weight(should_update=should_update_weight)
        ################################################
        ### 7. incrementaly trainning entity encoder ###
        ################################################
        if self.has_entity_encoder and budget == self.get_available_min_budget():
            X = np.array([self.budget2obvs[budget]["vectors"][-1]])
            y = np.array([self.budget2obvs[budget]["losses"][-1]])
            self.tpe_config_transformer.fit_encoder(X, y)
            if self.tpe_config_transformer.encoder.transform_matrix_status == "Updated":
                self.plot_entity_encoder_points()

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

    @ignore_warnings(category=UserWarning)
    def update_weight(self, should_update=0):
        # 判断是否要更新权重
        max_budget = self.get_available_max_budget()
        # fixme: 热启动时更新权重 : force_update
        if self.support_ensemble_epms and self.budget2epm[max_budget] is not None:
            available_budgets = self.get_available_budgets()
            if len(available_budgets) < 2:
                return
            n_exceed_points = (len(self.budget2obvs[max_budget]["losses"]) - self.min_points_in_model)
            if n_exceed_points < 0:
                return
            if should_update < 0:
                return
            if should_update > 0 or (n_exceed_points % self.update_weight_steps == 0):
                force_msg = "force" if should_update > 0 else ""
                self.logger.info(f"In max_budget {max_budget}, n_exceed_points = {n_exceed_points}, "
                                 f"{force_msg} updating EPMs' weight.")
                config_evaluator = self.budget2confevt[max_budget]
                configs = self.budget2obvs[max_budget]["configs"]
                losses = np.array(self.budget2obvs[max_budget]["losses"])
                X = self.transform(configs)
                #  调整各个epm的权重，使得对max_budget的update_weight_steps个观测的rank_loss最小
                # 从max_budget对应的losses中，按照分布分为update_weight_steps个bins，然后从每个bins中取一个样本
                # 取出的样本当做训练数据X_test, y_test, 剩下的用来训练max_budget_epm
                n_test_points = max(self.update_weight_accepted_samples, int(losses.size * self.ensemble_test_size))
                n_test_points = min(np.unique(losses).size, n_test_points)
                budget2weight_ = defaultdict(list)

                for i in range(10):
                    kbins = KBinsDiscretizer(n_test_points, strategy="quantile", encode="ordinal")
                    bins = kbins.fit_transform(losses[:, None]).flatten()
                    sampled_indexes = []
                    indexes = np.arange(len(losses), dtype="int")
                    # 每个bins中抽一个样本
                    for bin_id in np.unique(bins):
                        mask = bins == bin_id
                        masked_indexes = indexes[mask]
                        sampled_indexes.append(self.rng.choice(masked_indexes))
                    sampled_indexes = np.array(sampled_indexes)
                    # 现在， 已经均匀地从样本中抽了 n_test_points 个样本 ，抽出来的样本用来当EPMs的训练集，
                    # 剩下的样本用来训练max_busget_epm
                    rest_indexes = np.setdiff1d(indexes, sampled_indexes)
                    if len(sampled_indexes) < n_test_points:
                        n_additional_points = n_test_points - len(sampled_indexes)
                        additional_indexes = self.rng.choice(rest_indexes, size=n_additional_points, replace=False)
                        sampled_indexes = np.hstack([sampled_indexes, additional_indexes])
                        rest_indexes = np.setdiff1d(indexes, sampled_indexes)
                    losses_trans = self.loss_transformer.fit_transform(losses)
                    y_test = losses_trans[sampled_indexes]
                    X_test = X[sampled_indexes, :]
                    # max_budget_epm = deepcopy(self.epm)
                    # max_budget_epm.fit(X[rest_indexes, :], losses_trans[rest_indexes])
                    budget2weight = config_evaluator.calc_weight(X_test, y_test)
                    for b, w in budget2weight.items():
                        budget2weight_[b].append(w)
                budget2weight = {b: np.mean(ws) for b, ws in budget2weight_.items()}
                budgets = budget2weight.keys()
                weights = np.array(list(budget2weight.values()))
                weights /= weights.sum()
                budget2weight = dict(zip(budgets, weights))
                config_evaluator.update_weight(budget2weight)
                info = {
                    "budget2weight_": budget2weight_,
                    "budget2weight": budget2weight,
                }
                path = self.fs.join(self.weight_path, f"{self.update_weight_cnt}.json")
                self.update_weight_cnt += 1
                Path(path).write_text(json.dumps(info))  # numpy.float64 dump 居然不报错？

    def get_config(self, budget):
        # get max_budget
        # calc by budget2epm
        max_budget = self.get_available_max_budget()
        # initial points
        if self.initial_points is not None and self.initial_points_index < len(self.initial_points):
            while True:
                if self.initial_points_index >= len(self.initial_points):
                    break
                initial_point_dict = self.initial_points[self.initial_points_index]
                initial_point = Configuration(self.config_space, initial_point_dict)
                self.initial_points_index += 1
                initial_point.origin = "User Defined"
                if not self.is_config_exist(budget, initial_point):
                    self.logger.info(f"Using initial points [{self.initial_points_index - 1}]")
                    return self.process_config_info_pair(initial_point, {}, budget)
        # random sampling
        epm = self.budget2epm[max_budget]
        if epm is None:
            max_sample = 1000
            i = 0
            info_dict = {"model_based_pick": False}
            while i < max_sample:
                i += 1
                config = self.config_space.sample_configuration()
                add_configs_origin(config, "Initial Design")
                if self.is_config_exist(budget, config):
                    self.logger.info(f"The sample already exists and needs to be resampled. "
                                     f"It's the {i}-th time sampling in random sampling. ")
                else:
                    return self.process_config_info_pair(config, info_dict, budget)
            # todo: 收纳这个代码块
            seed = self.rng.randint(1, 8888)
            self.config_space.seed()
            config = self.config_space.sample_configuration()
            add_configs_origin(config, "Initial Design")
            info_dict.update({"sampling_different_samples_failed": True, "seed": seed})
            return self.process_config_info_pair(config, info_dict, budget)

        info_dict = {"model_based_pick": True}
        # thompson sampling
        if self.use_thompson_sampling > 0:
            hit = False
            if self.use_thompson_sampling >= 2:
                hit = True
            ts_config, ts_info_dict = self.thompson_sampling(max_budget, info_dict, hit)
            if ts_config is not None:
                self.logger.info("Using thompson sampling near the dominant samples.")
                return self.process_config_info_pair(ts_config, ts_info_dict, budget)
        # using config_evaluator evaluate random samples
        configs = self.config_space.sample_configuration(self.n_samples)
        losses, configs_sorted = self.evaluate(configs, max_budget, return_loss_config=True)
        add_configs_origin(configs_sorted, "Random Search (Sorted)")
        if self.use_local_search:
            start_points = self.get_local_search_initial_points(max_budget, 10, configs_sorted)
            local_losses, local_configs = self.local_search(start_points,
                                                            max_budget)
            add_configs_origin(local_configs, "Local Search")
            concat_losses = np.hstack([losses.flatten(), local_losses.flatten()])
            concat_configs = configs + local_configs
            random_var = self.rng.rand(len(concat_losses))
            indexes = np.lexsort((random_var.flatten(), concat_losses))
            concat_configs_sorted = [concat_configs[i] for i in indexes]
            concat_losses = concat_losses[indexes]
        else:
            concat_losses, concat_configs_sorted = losses, configs_sorted
        # 选取获益最大，且没有出现过的一个配置
        for i, config in enumerate(concat_configs_sorted):
            if self.is_config_exist(budget, config):
                self.logger.info(f"The sample already exists and needs to be resampled. "
                                 f"It's the {i}-th time sampling in bayesian sampling. ")
                # 超过 max_repeated_samples ， 用TS算法采样
                if i >= self.max_repeated_samples and self.use_thompson_sampling > 0:
                    ts_config, ts_info_dict = self.thompson_sampling(max_budget, info_dict, True)
                    return self.process_config_info_pair(ts_config, ts_info_dict, budget)
            else:
                return self.process_config_info_pair(config, info_dict, budget)
        # todo: 收纳这个代码块
        seed = self.rng.randint(1, 8888)
        self.config_space.seed(seed)
        config = self.config_space.sample_configuration()
        add_configs_origin(config, "Initial Design")
        info_dict.update({"sampling_different_samples_failed": True, "seed": seed})
        return self.process_config_info_pair(config, info_dict, budget)

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
        vectors_list = []  # TODO check
        budgets = [budget_ for budget_ in list(self.budget2obvs.keys()) if budget_ >= budget]
        for budget_ in budgets:
            vectors = np.array(self.budget2obvs[budget_]["locks"])
            if vectors.size:
                vectors_list.append(vectors)
        if len(vectors_list) == 0:
            return False
        vectors = np.vstack(vectors_list)
        if np.any(np.array(vectors.shape) == 0):
            return False
        vectors[np.isnan(vectors)] = -1
        vector = config.get_array().copy()
        vector[np.isnan(vector)] = -1
        if np.any(np.all(vector == vectors, axis=1)):
            return True
        return False

    def get_available_max_budget(self):
        budgets = [budget for budget in self.budget2epm.keys() if budget > 0]
        sorted_budgets = sorted(budgets)
        for budget in sorted(budgets, reverse=True):
            if budget <= 0:
                continue
            if self.budget2epm[budget] is not None:
                return budget
        return sorted_budgets[0]

    def get_available_min_budget(self):
        budgets = [budget for budget in self.budget2epm.keys() if budget > 0]
        for budget in sorted(budgets):
            if budget <= 0:
                continue
            return budget
        return None

    def get_available_budgets(self):
        available_budgets = []
        for budget in sorted(self.budget2epm.keys()):
            if self.budget2epm[budget] is not None:
                available_budgets.append(budget)
            else:
                break
        return available_budgets

    def thompson_sampling(self, budget, info_dict, hit=False) -> Tuple[Optional[Configuration], Optional[dict]]:
        # todo: 增加 n_candidates, sort_by_EI参数
        # On the call side, is max_budget
        info_dict = deepcopy(info_dict)
        rv = beta(self.alpha, self.beta)
        prob = rv.rvs(random_state=self.rng)
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
            kwargs = {
                "n_candidates": self.n_candidates,
                "sort_by_EI": self.sort_by_EI,
                "random_state": self.rng,
                "bandwidth_factor":self.bandwidth_factor
            }
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

    def process_config_info_pair(self, config: Configuration, info_dict: dict, budget):
        self.budget2obvs[budget]["locks"].append(config.get_array().copy())
        info_dict = deepcopy(info_dict)
        if config.origin is None:
            config.origin = "unknown"
        info_dict.update({
            "origin": config.origin
        })
        return config.get_dictionary(), info_dict

    def plot_entity_encoder_points(self):
        if not self.has_entity_encoder:
            return
        encoder = self.tpe_config_transformer.encoder
        n_plots = len(encoder.transform_matrix)
        n_rows = round(math.sqrt(n_plots))
        n_cols = math.ceil(n_plots / n_rows)
        plt.rcParams['figure.figsize'] = (6 * n_cols, 6 * n_rows)
        points_info = {}
        for i, (categories, col_name) in enumerate(zip(encoder.ordinal_encoder.categories_, encoder.cols)):
            points_info[col_name] = []
            tm = encoder.transform_matrix[i]
            if tm.shape[1] > 2:
                tm = TSNE(n_components=2).fit_transform(tm)
            elif tm.shape[1] < 2:
                self.logger.warning(f"col_name '{col_name}' dims < 2, continue.")
                continue
            choices = list(self.config_space.get_hyperparameter(col_name).choices)
            categories = categories.astype(int)
            choices = np.array(choices)[categories]
            if categories[0] < 0:
                choices[0] = "<NULL>"
            plt.subplot(n_rows, n_cols, i + 1)
            for j, choice in enumerate(choices):
                x = float(tm[j, 0])
                y = float(tm[j, 1])
                plt.scatter(x, y, label=choice)
                points_info[col_name].append({
                    "x": x,
                    "y": y,
                    "choice": choice,
                })
            plt.legend(loc='best')
            plt.title(col_name)
        json_path = self.fs.join(self.entity_path, f"{encoder.stage}.json")
        Path(json_path).write_text(json.dumps(points_info, indent=4))
        plt.suptitle(encoder.stage)
        plt.savefig(self.fs.join(self.entity_path, f"{encoder.stage}.svg"))
        plt.clf()
        plt.close()

    @property
    def has_entity_encoder(self):
        if self.has_entity_encoder_ is None:
            if self.use_thompson_sampling >= 0 and self.meta_encoder is not None:
                self.has_entity_encoder_ = True
            else:
                self.has_entity_encoder_ = False
        return self.has_entity_encoder_

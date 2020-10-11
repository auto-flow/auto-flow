import inspect
import multiprocessing as mp
import os
from collections import Counter
from copy import copy, deepcopy
from importlib import import_module
from pathlib import Path
from threading import Thread
from time import time
from typing import Union, Optional, Dict, Any, List, Type

import numpy as np
import pandas as pd
import psutil
from ConfigSpace import ConfigurationSpace
from frozendict import frozendict
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import autoflow.ambo.core.nameserver as hpns
from autoflow import constants
from autoflow.ambo.config_generators import ETBasedAMBO, RFBasedAMBO, GBRTBasedAMBO
from autoflow.ambo.result_logger import DatabaseResultLogger
from autoflow.ambo.utils import get_max_SH_iter, get_budgets, ConfigurationTransformer
from autoflow.constants import ExperimentType, SUBSAMPLES_BUDGET_MODE, PER_RUN_TIME_LIMIT, TIME_LEFT_FOR_THIS_TASK
from autoflow.data_container import DataFrameContainer
from autoflow.data_manager import DataManager
from autoflow.ensemble.mean.regressor import MeanRegressor
from autoflow.ensemble.trained_data_fetcher import TrainedDataFetcher
from autoflow.ensemble.trials_fetcher import TrialsFetcher
from autoflow.ensemble.vote.classifier import VoteClassifier
from autoflow.evaluation.budget import get_default_algo2iter, get_default_algo2budget_mode, get_default_algo2weight_mode
from autoflow.evaluation.strategy import parse_evaluation_strategy
from autoflow.evaluation.train_evaluator import TrainEvaluator
from autoflow.hdl.hdl2cs import HDL2CS
from autoflow.hdl.hdl_constructor import HDL_Constructor
from autoflow.metalearning.meta_encoder import train_meta_encoder
from autoflow.metalearning.metafeatures.calc import calculate_metafeatures
from autoflow.metalearning.util import load_metalearning_repository
from autoflow.metrics import r2, accuracy
from autoflow.metrics.util import score2loss
from autoflow.optimizer import Optimizer
from autoflow.resource_manager.base import ResourceManager
from autoflow.utils.config_space import replace_phps, config_regulation
from autoflow.utils.klass import instancing, get_valid_params_in_kwargs, set_if_not_None
from autoflow.utils.list_ import multiply_to_list
from autoflow.utils.logging_ import get_logger, setup_logger
from autoflow.utils.net import get_a_free_port
from autoflow.utils.packages import get_class_name_of_module


class AutoFlowEstimator(BaseEstimator):
    checked_mainTask = None

    def __init__(
            self,
            resource_manager: Union[ResourceManager, str] = None,
            hdl_constructor: Union[HDL_Constructor, None, dict] = None,
            min_budget: float = 1,
            max_budget: float = 1,
            eta: float = 4,
            SH_only: bool = True,
            algo2budget_mode: Optional[Dict[str, str]] = None,
            algo2weight_mode: Optional[Dict[str, str]] = None,
            algo2iter: Optional[Dict[str, int]] = None,
            specific_out_feature_groups_mapper: Dict[str, str] = frozendict({"encode.ordinal": "ordinal"}),
            only_use_subsamples_budget_mode: bool = False,
            k_folds: int = 5,
            holdout_test_size: float = 1 / 3,
            evaluation_strategy="simple",  # will rewrite k_folds, refit, min_budget, max_budget, eta
            SH_holdout_condition="s > 1e5 or (s * f > 1e6 and s > 1e3)",
            warm_start=True,
            config_generator: Union[str, Type] = "et-based-ambo",
            config_generator_params: dict = frozendict(),
            ns_host: str = "127.0.0.1",
            ns_port: int = 9090,
            worker_host: str = "127.0.0.1",
            master_host: str = "127.0.0.1",
            n_workers: int = 1,
            n_iterations: Optional[int] = None,
            min_n_workers: int = 1,
            concurrent_type: str = "process",
            model_registry: Dict[str, Type] = None,
            n_jobs_in_algorithm: Optional[int] = None,
            random_state: int = 42,
            log_path: str = "autoflow.log",
            log_config: Optional[dict] = None,
            highR_nan_threshold: float = 0.5,
            highC_cat_threshold: int = 4,
            consider_ordinal_as_cat: bool = False,
            should_store_intermediate_result: bool = False,
            refit: Optional[str] = "dynamic",
            fit_transformer_per_cv: bool = False,
            should_calc_all_metrics: bool = True,
            should_stack_X: bool = True,
            debug_evaluator: bool = False,
            initial_points=None,
            imbalance_threshold=2,
            per_run_time_limit=PER_RUN_TIME_LIMIT,
            time_left_for_this_task=TIME_LEFT_FOR_THIS_TASK,
            memory_limit=None,
            use_metalearning=True,
            mtl_path="~/autoflow/metalearning",
            mtl_init_method="kND",
            mtl_init_params=(5, 3),
            mtl_meta_epms=5,
            mtl_metafeatures=None,
            specific_hdl_rule="default",
            use_workflow_cache=True,
            **kwargs
    ):
        self.use_workflow_cache = use_workflow_cache
        self.mtl_init_method = mtl_init_method
        if algo2iter is None:
            algo2iter = get_default_algo2iter()
        if algo2weight_mode is None:
            algo2weight_mode = get_default_algo2weight_mode()
        if algo2budget_mode is None:
            algo2budget_mode = get_default_algo2budget_mode()
        if only_use_subsamples_budget_mode:
            algo2budget_mode = {key: SUBSAMPLES_BUDGET_MODE for key in algo2budget_mode}
        self.time_left_for_this_task = time_left_for_this_task
        self.specific_hdl_rule = specific_hdl_rule
        self.use_metalearning = use_metalearning
        self.mtl_meta_epms = mtl_meta_epms
        self.mtl_metafeatures = mtl_metafeatures
        self.mtl_init_params = mtl_init_params
        self.mtl_path = os.path.expandvars(os.path.expanduser(mtl_path))
        self.fit_transformer_per_cv = fit_transformer_per_cv
        self.SH_holdout_condition = SH_holdout_condition
        self.evaluation_strategy = evaluation_strategy
        self.per_run_time_limit = per_run_time_limit
        self.imbalance_threshold = imbalance_threshold
        self.initial_points = initial_points
        self.logger = get_logger(self)
        self.specific_out_feature_groups_mapper = specific_out_feature_groups_mapper
        self.warm_start = warm_start
        self.debug_evaluator = debug_evaluator
        self.only_use_subsamples_budget_mode = only_use_subsamples_budget_mode
        self.algo2iter = algo2iter
        self.algo2budget_mode = algo2budget_mode
        self.algo2weight_mode = algo2weight_mode
        self.config_generator_params = dict(config_generator_params)
        assert isinstance(k_folds, int) and k_folds >= 1
        # fixme: support int
        assert isinstance(holdout_test_size, float) and 0 < holdout_test_size < 1
        self.holdout_test_size = holdout_test_size
        self.k_folds = k_folds
        self.min_n_workers = min_n_workers
        self.master_host = master_host
        if n_jobs_in_algorithm is None:
            assert isinstance(n_workers, int) and n_workers >= 1, ValueError(f"Invalid n_workers {n_workers}")
            n_jobs_in_algorithm = int(np.clip(mp.cpu_count() // n_workers, 1, mp.cpu_count()))
            self.logger.info(f"`n_jobs_in_algorithm` is parsed to {n_jobs_in_algorithm}")
        # memory
        vm = psutil.virtual_memory()
        total = vm.total / 1024 / 1024
        free = vm.free / 1024 / 1024
        used = vm.used / 1024 / 1024
        self.free_memory = free
        self.logger.info(f"Computer's Memory Info: total = {total:.2f}M, free = {free:.2f}M, used = {used:.2f}M")
        if memory_limit is None:
            self.logger.info(
                "PER_RUN_MEMORY_LIMIT is None, will calc per_run_memory_limit by 'total / search_thread_num'")
            memory_limit = total / n_workers
        self.logger.info(f"memory_limit = {memory_limit}M")
        self.memory_limit = memory_limit
        # end memory
        self.n_jobs_in_algorithm = n_jobs_in_algorithm
        self.n_iterations = n_iterations
        self.concurrent_type = concurrent_type
        self.n_workers = n_workers
        self.worker_host = worker_host
        self.ns_port = ns_port
        self.ns_host = ns_host
        self.config_generator = config_generator
        self.SH_only = SH_only
        self.eta = eta
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.should_stack_X = should_stack_X
        self.consider_ordinal_as_cat = consider_ordinal_as_cat
        if model_registry is None:
            model_registry = {}
        assert isinstance(model_registry, dict)
        for key, value in model_registry.items():
            assert inspect.isclass(value)
        self.model_registry = model_registry
        self.refit = refit
        self.should_store_intermediate_result = should_store_intermediate_result
        self.should_calc_all_metrics = should_calc_all_metrics
        self.log_config = log_config
        self.highR_nan_threshold = highR_nan_threshold
        self.highC_cat_threshold = highC_cat_threshold
        # ---logger------------------------------------
        self.log_path = os.path.expandvars(os.path.expanduser(log_path))
        setup_logger(self.log_path, self.log_config)
        # ---random_state-----------------------------------
        self.random_state = random_state
        # ---hdl_constructor--------------------------
        self.hdl_constructor = instancing(hdl_constructor, HDL_Constructor, kwargs)
        # ---resource_manager-----------------------------------
        self.resource_manager: ResourceManager = instancing(resource_manager, ResourceManager, kwargs)
        # ---metalearning file structure------------------------
        fs = self.resource_manager.file_system
        fs.mkdir(self.mtl_path)
        self.mtl_repo_path = fs.join(self.mtl_path, "repository")
        fs.mkdir(self.mtl_repo_path)
        self.mtl_metafeatures_path = fs.join(self.mtl_path, "metafeatures.csv")
        # todo: generalize to s3, hdfs
        if fs.exists(self.mtl_metafeatures_path):
            origin_mtl_metafeatures = pd.read_csv(self.mtl_metafeatures_path, index_col=0)
        else:
            origin_mtl_metafeatures = pd.DataFrame()
        if self.mtl_metafeatures is None and origin_mtl_metafeatures.shape[0] == 0:
            import autoflow.metalearning as mtl_module
            mtl_metafeatures_csv = Path(mtl_module.__file__).parent / "metafeatures.csv"
            self.logger.info(f"Initially loading mtl_metafeatures_csv: '{mtl_metafeatures_csv}'")
            origin_mtl_metafeatures = pd.read_csv(mtl_metafeatures_csv, index_col=0)
            origin_mtl_metafeatures.to_csv(self.mtl_metafeatures_path)
        # todo: 设计一种机制，如果相比较原来的有更新，就写入原来的
        self.mtl_metafeatures_ = origin_mtl_metafeatures
        self.config_generator_mapper = {
            "et-based-ambo": ETBasedAMBO,
            "rf-based-ambo": RFBasedAMBO,
            "gbrt-based-ambo": GBRTBasedAMBO,
        }
        # ---member_variable------------------------------------
        self.estimator = None
        self.ensemble_estimator = None
        self.evaluators = []
        self.data_manager = None
        self.NS = None
        self.optimizer = None

    def get_holdout_splitter(self):
        # fix random_state to prevent change of task_id
        if self.ml_task.mainTask == "classification":
            return StratifiedShuffleSplit(n_splits=1, test_size=self.holdout_test_size, random_state=0)
        else:
            return ShuffleSplit(n_splits=1, test_size=self.holdout_test_size, random_state=0)

    def get_cv_splitter(self):
        if self.ml_task.mainTask == "classification":
            return StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=0)
        else:
            return KFold(n_splits=self.k_folds, shuffle=True, random_state=0)

    def hdl2configSpce(self, hdl: Dict):
        hdl2cs = HDL2CS()
        hdl2cs.set_task(self.ml_task)
        return hdl2cs(hdl)

    def update_evaluation_strategy(self, dict_: dict):
        for k, v in dict_.items():
            if v is not None:
                pv = getattr(self, k)
                if pv != v:
                    self.logger.info(f"update evaluation_strategy | {k}: {pv} -> {v}")
                    setattr(self, k, v)

    def input_experiment_data(
            self,
            X_train: Union[np.ndarray, pd.DataFrame, DataFrameContainer, str],
            y_train=None,
            X_test: Union[np.ndarray, pd.DataFrame, DataFrameContainer, str] = None,
            y_test=None,
            groups=None,
            upload_type="fs",
            sub_sample_indexes=None,
            sub_feature_indexes=None,
            column_descriptions: Optional[Dict] = frozendict(),
            metric=None,
            splitter=None,
            specific_task_token="",
            dataset_metadata: dict = frozenset(),
            task_metadata: dict = frozendict(),

    ):
        self.upload_type = upload_type
        self.sub_sample_indexes = sub_sample_indexes
        self.sub_feature_indexes = sub_feature_indexes
        dataset_metadata = dict(dataset_metadata)
        task_metadata = dict(task_metadata)
        column_descriptions = dict(column_descriptions)
        # build data_manager
        self.data_manager: DataManager = DataManager(
            self.resource_manager,
            X_train, y_train, X_test, y_test, dataset_metadata, column_descriptions, self.highR_nan_threshold,
            self.highC_cat_threshold, self.consider_ordinal_as_cat, upload_type
        )
        # parse ml_task
        self.ml_task = self.data_manager.ml_task
        if self.checked_mainTask is not None:
            if self.checked_mainTask != self.ml_task.mainTask:
                if self.checked_mainTask == "regression":
                    self.ml_task = constants.regression_task
                    self.data_manager.ml_task = self.ml_task
                else:
                    self.logger.error(
                        f"This task is supposed to be {self.checked_mainTask} task ,but the target data is {self.ml_task}.")
                    raise ValueError
        if self.ml_task.mainTask == "regression":
            self.n_classes = 1
        else:
            self.n_classes = pd.Series(self.data_manager.y_train).nunique()
        self.n_samples, self.n_features = X_train.shape
        # evaluation strategy
        if self.evaluation_strategy is not None:
            dict_ = parse_evaluation_strategy(self.evaluation_strategy, self.SH_holdout_condition,
                                              self.n_samples, self.n_features, self.n_classes, self.ml_task)
            self.update_evaluation_strategy(dict_)
        # parse splitter
        self.groups = groups
        self.n_samples = self.data_manager.X_train.shape[0]
        if splitter is None:
            if self.k_folds == 1:
                splitter = self.get_holdout_splitter()
            else:
                splitter = self.get_cv_splitter()
            self.logger.info(f"splitter is parsed to `{splitter}`")
        assert hasattr(splitter, "split") and hasattr(splitter, "n_splits"), \
            "Parameter 'splitter' should be a train-valid splitter, " \
            "which contain 'split(X, y, groups)' method, 'n_splits' attribute."
        if getattr(splitter, "random_state") is None:
            self.logger.warning(
                f"splitter '{splitter}' haven't specific random_state, it's random_state is set default '{self.random_state}'")
            splitter.random_state = self.random_state
        if self.ml_task.mainTask == "classification":
            X_train = self.data_manager.X_train.data
            y_train = self.data_manager.y_train.data
            for fold_ix, (train_ix, valid_ix) in enumerate(splitter.split(X_train, y_train)):
                self.logger.info(f"fold-{fold_ix} | y_train count = {dict(Counter(y_train[train_ix]))}")
                self.logger.info(f"fold-{fold_ix} | y_valid count = {dict(Counter(y_train[valid_ix]))}")
                # from autoflow.estimator.wrap_lightgbm import LGBMClassifier
                # lgbm=LGBMClassifier().fit(X_train.iloc[train_ix,:], y_train[train_ix], X_train.iloc[valid_ix,:], y_train[valid_ix])
                # score=lgbm.score(X_train.iloc[valid_ix,:], y_train[valid_ix])
                # print(score)
        self.n_splits = splitter.n_splits
        self.splitter = splitter
        # todo  set n_iterations
        # fixme: 有拍脑袋的嫌疑
        if self.n_iterations is None:
            if self.min_budget != self.max_budget:
                max_SH_iter = get_max_SH_iter(self.min_budget, self.max_budget, self.eta)
                if self.max_budget > 1:
                    self.n_iterations = max_SH_iter * 1
                else:
                    self.n_iterations = max_SH_iter * 1
            else:
                self.n_iterations = 50
        # parse metric
        if metric is None:
            if self.ml_task.mainTask == "regression":
                metric = r2
            elif self.ml_task.mainTask == "classification":
                # todo 多分类用f1_macro
                metric = accuracy
            else:
                raise NotImplementedError()
        self.metric = metric
        # get task_id, and insert record into "tasks.tasks" database
        self.resource_manager.insert_task_record(
            data_manager=self.data_manager, metric=metric, splitter=splitter,
            specific_task_token=specific_task_token, dataset_metadata=dataset_metadata, task_metadata=task_metadata,
            sub_sample_indexes=sub_sample_indexes, sub_feature_indexes=sub_feature_indexes)
        self.resource_manager.close_task_table()
        # store other params
        self.hdl_constructor.run(self.data_manager, self.model_registry, self.imbalance_threshold,
                                 self.specific_hdl_rule)
        self.hdl = self.hdl_constructor.get_hdl()
        self.config_space: ConfigurationSpace = self.hdl2configSpce(self.hdl)
        self.config_space.seed(self.random_state)
        if self.n_jobs_in_algorithm is not None:
            replace_phps(self.config_space, "n_jobs", int(self.n_jobs_in_algorithm))
        # get hdl_id, and insert record into "{task_id}.hdls" database
        self.resource_manager.insert_hdl_record(self.hdl, self.hdl_constructor.hdl_metadata)
        self.resource_manager.close_hdl_table()
        self.resource_manager.insert_budget_record({
            "algo2budget_mode": self.algo2budget_mode,
            "algo2iter": self.algo2iter,
            "min_budget": self.min_budget,
            "max_budget": self.max_budget,
            "eta": self.eta,
        })
        self.task_id = self.resource_manager.task_id
        self.hdl_id = self.resource_manager.hdl_id
        self.user_id = self.resource_manager.user_id
        self.budget_id = self.resource_manager.budget_id
        self.logger.info(f"task_id:\t{self.task_id}")
        self.logger.info(f"hdl_id:\t{self.hdl_id}")
        self.logger.info(f"budget_id:\t{self.budget_id}")
        self.run_id = f"{self.task_id}-{self.hdl_id}-{self.user_id}"
        self.worker_cnt = 0
        #
        if isinstance(self.initial_points, list):
            self.initial_points, _ = config_regulation(self.config_space, self.initial_points)

    def insert_experiment_record(
            self,
            additional_info: dict = frozendict(),
            fit_ensemble_params: Union[str, Dict[str, Any], None, bool] = "auto",

    ):
        additional_info = dict(additional_info)
        # now we get task_id and hdl_id, we can insert current runtime information into "experiments.experiments" database
        experiment_config = {
            "should_stack_X": self.should_stack_X,
            "refit": self.refit,
            "should_calc_all_metric": self.should_calc_all_metrics,
            "should_store_intermediate_result": self.should_store_intermediate_result,
            "fit_ensemble_params": str(fit_ensemble_params),
            "highR_nan_threshold": self.highR_nan_threshold,
            "highC_cat_threshold": self.highC_cat_threshold,
            "consider_ordinal_as_cat": self.consider_ordinal_as_cat,
            "random_state": self.random_state,
            "log_path": self.log_path,
            "log_config": self.log_config,
        }
        # fixme
        is_manual = False
        if is_manual:
            experiment_type = ExperimentType.MANUAL
        else:
            experiment_type = ExperimentType.AUTO
        self.resource_manager.insert_experiment_record(experiment_type, experiment_config, additional_info)
        self.resource_manager.close_experiment_table()
        self.experiment_id = self.resource_manager.experiment_id
        self.logger.info(f"experiment_id:\t{self.experiment_id}")

    def run_nameserver(self, ns_host=None, ns_port=None):
        set_if_not_None(self, "ns_host", ns_host)
        set_if_not_None(self, "ns_port", ns_port)
        self.ns_port = get_a_free_port(self.ns_port, self.ns_host)
        self.NS = hpns.NameServer(run_id=self.run_id, host=self.ns_host, port=self.ns_port)
        self.NS.start()
        self.ns_host = self.NS.host
        self.ns_port = self.NS.port

    def run_evaluator(self, worker_host, background, concurrent_type, worker_id=None):
        if worker_id is None:
            worker_id = self.worker_cnt
        evaluator = TrainEvaluator(
            run_id=self.run_id,
            data_manager=self.data_manager,
            resource_manager=self.resource_manager,
            random_state=self.random_state,
            metric=self.metric,
            groups=self.groups,
            should_calc_all_metric=self.should_calc_all_metrics,
            splitter=self.splitter,
            should_store_intermediate_result=self.should_store_intermediate_result,
            should_stack_X=self.should_stack_X,
            refit=self.refit,
            fit_transformer_per_cv=self.fit_transformer_per_cv,
            model_registry=self.model_registry,
            algo2budget_mode=self.algo2budget_mode,
            algo2weight_mode=self.algo2weight_mode,
            algo2iter=self.algo2iter,
            only_use_subsamples_budget_mode=self.only_use_subsamples_budget_mode,
            specific_out_feature_groups_mapper=self.specific_out_feature_groups_mapper,
            max_budget=self.max_budget,
            nameserver=self.ns_host,
            nameserver_port=self.ns_port,
            host=worker_host,
            worker_id=worker_id,
            timeout=None,
            debug=self.debug_evaluator,
            per_run_time_limit=self.per_run_time_limit,
            memory_limit=self.memory_limit,
            use_workflow_cache=self.use_workflow_cache,
            insert_trial_table=True
        )
        self.worker_cnt += 1
        self.evaluators.append(evaluator)
        evaluator.run(background, concurrent_type)

    def run_evaluators(
            self,
            worker_host: Union[str, None, List[str]] = None,
            background: Union[str, None, List[str]] = None,
            concurrent_type: Union[str, None, List[str]] = None,
            worker_id: Union[None, List[Union[str, int]]] = None,
            n_workers: Optional[int] = None
    ):
        if worker_host is None:
            worker_host = self.worker_host
        if background is None:
            background = True
        if concurrent_type is None:
            concurrent_type = self.concurrent_type
        if isinstance(worker_id, (int, str)):
            worker_id = None
        if n_workers is None:
            n_workers = self.n_workers
        worker_host = multiply_to_list(worker_host, n_workers)
        background = multiply_to_list(background, n_workers)
        concurrent_type = multiply_to_list(concurrent_type, n_workers)
        worker_id = multiply_to_list(worker_id, n_workers)
        for worker_host_, background_, concurrent_type_, worker_id_ in zip(
                worker_host, background, concurrent_type, worker_id
        ):
            self.run_evaluator(worker_host_, background_, concurrent_type_, worker_id_)

    # Semantic compatibility with ambo
    run_workers = run_evaluators

    def run_optimizer(
            self, master_host=None, min_budget=None, max_budget=None, eta=None,
            config_generator=None, config_generator_param=None, n_iterations=None,
            min_n_workers=None
    ):
        set_if_not_None(self, "master_host", master_host)
        set_if_not_None(self, "min_budget", min_budget)
        set_if_not_None(self, "max_budget", max_budget)
        set_if_not_None(self, "eta", eta)
        set_if_not_None(self, "config_generator", config_generator)
        set_if_not_None(self, "config_generator_param", config_generator_param)
        set_if_not_None(self, "n_iterations", n_iterations)
        set_if_not_None(self, "min_n_workers", min_n_workers)
        # name to class
        if inspect.isclass(self.config_generator):
            self.logger.info(f"'{self.config_generator}' is a class from '{self.config_generator.__module__}' module.")
            cg_cls = self.config_generator
        elif isinstance(self.config_generator, str):
            if self.config_generator in self.config_generator_mapper:
                cg_cls = self.config_generator_mapper[self.config_generator]
            else:
                module = import_module("autoflow.ambo.config_generators")
                cg_cls = getattr(module, self.config_generator)
        else:
            raise NotImplementedError
        # instancing config_generator
        budgets = get_budgets(self.min_budget, self.max_budget, self.eta)
        if self.use_metalearning:
            self.config_generator_params.update(meta_encoder=self.config_transformer.encoder)
        fs = self.resource_manager.file_system
        rm = self.resource_manager
        if "record_path" not in self.config_generator_params:
            record_path = fs.join(rm.experiment_path, "ambo")
            self.config_generator_params["record_path"] = record_path
        config_generator_ = cg_cls(
            self.config_space, budgets, self.random_state, self.initial_points, fs,
            **self.config_generator_params)
        if self.use_metalearning:
            # plot entity_encoder
            config_generator_.plot_entity_encoder_points()
            # training meta epms
            start_time = time()
            meta_epms = {}
            smac_config_transformer = ConfigurationTransformer(impute=-1, encoder=None).fit(self.config_space)
            for i in range(self.mtl_meta_epms):
                epm = copy(config_generator_.epm)
                X = smac_config_transformer.transform(self.ND2obvs[self.k_nearest_dataset_ids[i]]["vectors"])
                y = config_generator_.loss_transformer.fit_transform(
                    self.ND2obvs[self.k_nearest_dataset_ids[i]]["losses"])
                meta_epms[-i] = epm.fit(X, y)
            config_generator_.budget2epm.update(meta_epms)
            cost_time = time() - start_time
            self.logger.info(f"training meta epms cost {cost_time:.3f}s")
        # database_result_logger
        self.database_result_logger = DatabaseResultLogger(self.resource_manager)
        # warm start
        if self.warm_start:
            previous_result, incumbents, incumbent_performances = self.resource_manager.get_result_from_trial_table(
                task_id=self.task_id,
                hdl_id=self.hdl_id,
                user_id=self.user_id,
                budget_id=self.budget_id,
                config_space=self.config_space
            )
        else:
            previous_result, incumbents, incumbent_performances = None, None, None
        # instancing optimizer
        self.optimizer = Optimizer(
            self.run_id,
            config_generator_,
            time_left_for_this_task=self.time_left_for_this_task,
            nameserver=self.ns_host,
            nameserver_port=self.ns_port,
            host=self.master_host,
            result_logger=self.database_result_logger,
            previous_result=previous_result,
            min_budget=self.min_budget,
            max_budget=self.max_budget,
            eta=self.eta,
            SH_only=self.SH_only,
            incumbents=incumbents,
            incumbent_performances=incumbent_performances
        )
        # run
        self.opt_result = self.optimizer.run(self.n_iterations, self.min_n_workers)

    # Semantic compatibility with ambo
    run_master = run_optimizer

    def run_metalearning(self):
        if not self.use_metalearning:
            return
        # calculate metafeatures
        metafeatures: dict = calculate_metafeatures(
            self.data_manager.feature_groups,
            self.data_manager.X_train.data.values,
            self.data_manager.y_train.data,
            self.ml_task,
            self.free_memory
        )
        metafeatures_ = pd.Series(metafeatures)
        # finding k nearest neighbors
        mtl_metafeatures_ = self.mtl_metafeatures_[metafeatures_.index].values
        metafeatures_ = metafeatures_.values.reshape(1, -1)
        scaler = MinMaxScaler()
        scaler.fit(mtl_metafeatures_)
        mtl_metafeatures_ = scaler.transform(mtl_metafeatures_)
        metafeatures_ = scaler.transform(metafeatures_)
        pds = pairwise_distances(metafeatures_, mtl_metafeatures_, metric="l1").flatten()
        top_m = 1
        if self.mtl_init_method == "kND":
            assert len(self.mtl_init_params) == 2, \
                ValueError("first item is 'top-k' nearest datasets, second item is 'top-m' best configs.")
            top_k, top_m = self.mtl_init_params
            if top_m > 1:
                assert top_k <= self.mtl_meta_epms, \
                    ValueError("when 'top-m' > 1, 'top-k' should <= mtl_meta_epms.")
        else:
            top_k = self.mtl_meta_epms
        indexes = pds.argsort()[:top_k]
        self.k_nearest_dataset_ids = self.mtl_metafeatures_.iloc[indexes, :].index.tolist()
        # assemble data
        # config regulation
        start_time = time()
        fs = self.resource_manager.file_system
        # fixme: when fs is hdfs
        results = Parallel(backend="threading", n_jobs=mp.cpu_count())(
            delayed(load_metalearning_repository)(fs, self.mtl_repo_path, nearest_dataset_id)
            for nearest_dataset_id in self.k_nearest_dataset_ids
        )
        ND2repo = {nearest_dataset_id: result
                   for nearest_dataset_id, result in zip(self.k_nearest_dataset_ids, results)}
        ND2obvs = {}
        metric_name = self.metric.name
        if self.concurrent_type == "process":
            meta_encoder_results = mp.Manager().list()
        else:
            meta_encoder_results = []
        obj = None
        for i, (dataset_id, repo) in enumerate(ND2repo.items()):
            ND2obvs[dataset_id] = {"configs": [], "losses": []}
            is_meta_epms = i < self.mtl_meta_epms
            if metric_name in repo[0][1]:
                picked_metric_name = metric_name
            else:
                picked_metric_name = "accuracy"
            if is_meta_epms:
                for row in repo:
                    ND2obvs[dataset_id]["configs"].append(row[0])
                    ND2obvs[dataset_id]["losses"].append(
                        score2loss(row[1][picked_metric_name], self.metric))
            else:
                # fixme: 如果最好的配置是非法的，会报错
                losses = [score2loss(row[1][picked_metric_name], self.metric) for row in repo]
                best_idx = int(np.argmin(losses))
                ND2obvs[dataset_id]["configs"].append(repo[best_idx][0])
                ND2obvs[dataset_id]["losses"].append(losses[best_idx])
            configs, vectors, activates = config_regulation(self.config_space, ND2obvs[dataset_id]["configs"],
                                                            random_state=self.random_state, return_activate=True)
            ND2obvs[dataset_id]["configs_reg"] = configs
            ND2obvs[dataset_id]["vectors"] = vectors
            activates = np.array(activates)
            losses = np.array(ND2obvs[dataset_id]["losses"])[activates]
            ND2obvs[dataset_id]["activates"] = activates
            ND2obvs[dataset_id]["losses"] = losses
            if i == 0:  # todo: add option support close meta_encoder
                encoder_params = dict(max_epoch=100, early_stopping_rounds=50, n_jobs=1, verbose=0)
                X = ND2obvs[self.k_nearest_dataset_ids[0]]['vectors']
                y = ND2obvs[self.k_nearest_dataset_ids[0]]['losses']
                args = (encoder_params, self.config_space, X, y, meta_encoder_results)
                if self.concurrent_type == "process":
                    obj = mp.Process(target=train_meta_encoder, args=args)
                elif self.concurrent_type == "thread":
                    obj = Thread(target=train_meta_encoder, args=args)
                else:
                    raise ValueError(f"Unknown concurrent_type {self.concurrent_type}")
                obj.start()
        suggestions = []
        # todo: add to options
        discrepant_variables = ["estimating:__choice__", "preprocessing:normed->final:__choice__"]
        for nearest_dataset_id in self.k_nearest_dataset_ids:
            obvs = ND2obvs[nearest_dataset_id]
            existing_variables = set()
            for idx in np.argsort(obvs["losses"]):
                config = obvs["configs_reg"][idx]
                variables = tuple([config.get(var) for var in discrepant_variables])
                if variables not in existing_variables:
                    existing_variables.add(variables)
                    suggestions.append(config)
                if len(existing_variables) >= top_m:
                    break
        if obj is not None:
            obj.join()
            self.config_transformer = deepcopy(meta_encoder_results[0])
        cost_time = time() - start_time
        self.logger.info(f"config_regulation and train_meta_encoder cost {cost_time:.3f}s")
        self.ND2obvs = ND2obvs
        if not isinstance(self.initial_points, list):
            self.initial_points = []
        self.initial_points.extend(suggestions)

    def fit(
            self,
            X_train: Union[np.ndarray, pd.DataFrame, DataFrameContainer, str],
            y_train=None,
            X_test: Union[np.ndarray, pd.DataFrame, DataFrameContainer, str] = None,
            y_test=None,
            groups=None,
            upload_type="fs",
            sub_sample_indexes=None,
            sub_feature_indexes=None,
            column_descriptions: Optional[Dict] = frozendict(),
            metric=None,
            splitter=None,
            specific_task_token="",
            dataset_metadata: dict = frozenset(),
            task_metadata: dict = frozendict(),
            additional_info: dict = frozendict(),
            fit_ensemble_params: Union[str, Dict[str, Any], None, bool] = "auto",
            is_not_realy_run=False,
    ):
        '''

        Parameters
        ----------
        X_train: :class:`numpy.ndarray` or :class:`pandas.DataFrame`
        y_train: :class:`numpy.ndarray` or :class:`pandas.Series` or str
        X_test: :class:`numpy.ndarray` or :class:`pandas.DataFrame` or None
        y_test: :class:`numpy.ndarray` or :class:`pandas.Series` or str
        column_descriptions: dict
            Description about each columns' feature_group, you can find full definition in :class:`autoflow.manager.data_manager.DataManager` .
        dataset_metadata: dict
            Dataset's metadata
        metric: :class:`autoflow.metrics.Scorer` or None
            If ``metric`` is None:

            if it's classification task, :obj:`autoflow.metrics.accuracy` will be used by default.

            if it's regressor task, :obj:`autoflow.metrics.r2` will be used by default.
        should_calc_all_metrics: bool
            If ``True``, all the metrics supported in current task will be calculated, result will be store in databbase.
        splitter: object
            Default is ``KFold(5, True, 42)`` object. You can pass this param defined by yourself or other package,
            like :class:`sklearn.model_selection.StratifiedKFold`.
        specific_task_token: str
        should_store_intermediate_result: bool
        additional_info: dict
        fit_ensemble_params: str, dict, None, bool
            If this param is None, program will not do ensemble.

            If this param is "auto" or True, the top 10 models will be integrated by stacking ensemble.
        Returns
        -------
        self
        '''
        setup_logger(self.log_path, self.log_config)
        self.input_experiment_data(
            X_train, y_train, X_test, y_test, groups, upload_type, sub_sample_indexes,
            sub_feature_indexes, column_descriptions, metric, splitter, specific_task_token,
            dataset_metadata, task_metadata
        )
        if is_not_realy_run:
            return self
        self.insert_experiment_record(
            additional_info, fit_ensemble_params
        )
        self.run_metalearning()
        self.run_nameserver()
        self.run_evaluators()
        self.run_optimizer()
        self.optimizer.shutdown(shutdown_workers=True)
        self.NS.shutdown()
        savedpath = os.getenv("SAVEDPATH")
        if savedpath is not None:
            os.system(f"rm -rf {savedpath}/autoflow")
        else:
            self.start_final_step(fit_ensemble_params)
        self.resource_manager.finish_experiment(self.log_path, self)
        return self
        # todo: 重新实现手动建模

    def start_final_step(self, fit_ensemble_params):
        if isinstance(fit_ensemble_params, str):
            if fit_ensemble_params == "auto":
                self.logger.info(f"'fit_ensemble_params' is 'auto', use default params to fit_ensemble_params.")
                self.estimator = self.fit_ensemble(fit_ensemble_alone=False)
            else:
                raise NotImplementedError
        elif isinstance(fit_ensemble_params, bool):
            if fit_ensemble_params:
                self.logger.info(f"'fit_ensemble_params' is True, use default params to fit_ensemble_params.")
                self.estimator = self.fit_ensemble(fit_ensemble_alone=False)
            else:
                self.logger.info(
                    f"'fit_ensemble_params' is False, don't fit_ensemble but use best trial as result.")
                self.estimator = self.resource_manager.load_best_estimator(self.ml_task)
        elif isinstance(fit_ensemble_params, dict):
            self.logger.info(
                f"'fit_ensemble_params' is specific: {fit_ensemble_params}.")
            self.estimator = self.fit_ensemble(fit_ensemble_alone=False, **fit_ensemble_params)
        elif fit_ensemble_params is None:
            self.logger.info(
                f"'fit_ensemble_params' is None, don't fit_ensemble but use best trial as result.")
            self.estimator = self.resource_manager.load_best_estimator(self.ml_task)
        else:
            raise NotImplementedError

    def get_evaluator_params(self, random_state=None, resource_manager=None):
        if resource_manager is None:
            resource_manager = self.resource_manager
        if random_state is None:
            random_state = self.random_state
        if not hasattr(self, "instance_id"):
            self.instance_id = ""
            self.logger.warning(f"{self.__class__.__name__} haven't 'instance_id'!")
        return dict(
            random_state=random_state,
            data_manager=self.data_manager,
            metric=self.metric,
            groups=self.groups,
            should_calc_all_metric=self.should_calc_all_metrics,
            splitter=self.splitter,
            should_store_intermediate_result=self.should_store_intermediate_result,
            should_stack_X=self.should_stack_X,
            resource_manager=resource_manager,
            refit=self.refit,
            model_registry=self.model_registry,
            instance_id=self.instance_id
        )

    def fit_ensemble(
            self,
            task_id=None,
            hdl_id=None,
            trials_fetcher_cls="GetBestK",
            trials_fetcher_params=frozendict(k=10),
            ensemble_type="stack",
            ensemble_params=frozendict(),
            fit_ensemble_alone=True
    ):
        # fixme: ensemble_params可能会面临一个问题，就是传入无法序列化的内容
        trials_fetcher_params = dict(trials_fetcher_params)
        ensemble_params = dict(ensemble_params)
        kwargs = get_valid_params_in_kwargs(self.fit_ensemble, locals())
        if task_id is None:
            assert hasattr(self.resource_manager, "task_id") and self.resource_manager.task_id is not None
            task_id = self.resource_manager.task_id
        self.task_id = task_id
        self.resource_manager.task_id = task_id
        if hdl_id is not None:
            self.hdl_id = hdl_id
            self.resource_manager.hdl_id = hdl_id
        if fit_ensemble_alone:
            setup_logger(self.log_path, self.log_config)
            if fit_ensemble_alone:
                experiment_config = {
                    "fit_ensemble_params": kwargs
                }
                self.resource_manager.insert_experiment_record(ExperimentType.ENSEMBLE, experiment_config, {})
                self.experiment_id = self.resource_manager.experiment_id
        from autoflow.ensemble import trials_fetcher
        assert hasattr(trials_fetcher, trials_fetcher_cls)
        trials_fetcher_cls = getattr(trials_fetcher, trials_fetcher_cls)
        trials_fetcher_inst: TrialsFetcher = trials_fetcher_cls(
            resource_manager=self.resource_manager,
            task_id=task_id,
            hdl_id=hdl_id,
            **trials_fetcher_params
        )
        trial_ids = trials_fetcher_inst.fetch()
        estimator_list, y_true_indexes_list, y_preds_list = TrainedDataFetcher(
            task_id, hdl_id, trial_ids, self.resource_manager).fetch()
        # todo: 在这里，只取了验证集的数据，没有取测试集的数据。待拓展
        ml_task, y_true = self.resource_manager.get_ensemble_needed_info(task_id)
        if len(estimator_list) == 0:
            raise ValueError("Length of estimator_list must >=1. ")
        elif len(estimator_list) == 1:
            self.logger.info("Length of estimator_list == 1, don't do ensemble.")
            if ml_task.mainTask == "classification":
                ensemble_estimator = VoteClassifier(estimator_list[0])
            else:
                ensemble_estimator = MeanRegressor(estimator_list[0])
        else:
            ensemble_estimator_package_name = f"autoflow.ensemble.{ensemble_type}.{ml_task.role}"
            ensemble_estimator_package = import_module(ensemble_estimator_package_name)
            ensemble_estimator_class_name = get_class_name_of_module(ensemble_estimator_package_name)
            ensemble_estimator_class = getattr(ensemble_estimator_package, ensemble_estimator_class_name)
            # ensemble_estimator : EnsembleEstimator
            ensemble_estimator = ensemble_estimator_class(**ensemble_params)
            ensemble_estimator.fit_trained_data(estimator_list, y_true_indexes_list, y_preds_list, y_true)
        self.ensemble_estimator = ensemble_estimator
        if fit_ensemble_alone:
            self.estimator = self.ensemble_estimator
            self.resource_manager.finish_experiment(self.log_path, self)
        return self.ensemble_estimator

    def auto_fit_ensemble(self):
        # todo: 调研stacking等ensemble方法的表现评估
        pass

    def _predict(
            self,
            X_test: Union[DataFrameContainer, pd.DataFrame, np.ndarray],
    ):
        self.data_manager.set_data(X_test=X_test)

    def copy(self):
        tmp_dm = self.data_manager
        tmp_NS = self.NS
        tmp_evaluators = self.evaluators
        tmp_optimizer = self.optimizer
        self.NS = None
        self.evaluators = None
        self.optimizer = None
        self.data_manager: DataManager = self.data_manager.copy(
            keep_data=False) if self.data_manager is not None else None
        self.resource_manager.start_safe_close()
        res = copy(self)
        self.resource_manager.end_safe_close()
        self.NS = tmp_NS
        self.evaluators = tmp_evaluators
        self.optimizer = tmp_optimizer
        self.data_manager: DataManager = tmp_dm
        return res

    def pickle(self):
        # todo: 怎么做保证不触发self.resource_manager的__reduce__
        from pickle import dumps
        tmp_dm = self.data_manager
        tmp_NS = self.NS
        tmp_evaluators = self.evaluators
        tmp_optimizer = self.optimizer
        self.NS = None
        self.evaluators = None
        self.optimizer = None
        self.data_manager: DataManager = self.data_manager.copy(
            keep_data=False) if self.data_manager is not None else None
        self.resource_manager.start_safe_close()
        res = dumps(self)
        self.resource_manager.end_safe_close()
        self.NS = tmp_NS
        self.evaluators = tmp_evaluators
        self.optimizer = tmp_optimizer
        self.data_manager: DataManager = tmp_dm
        return res

import inspect
import os
from copy import deepcopy
from importlib import import_module
from typing import Union, Optional, Dict, Any, List, Type

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from frozendict import frozendict
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit

import autoflow.opt.core.nameserver as hpns
from autoflow import constants
from autoflow.constants import ExperimentType, SUBSAMPLES_BUDGET_MODE
from autoflow.data_container import DataFrameContainer
from autoflow.data_manager import DataManager
from autoflow.ensemble.base import EnsembleEstimator
from autoflow.ensemble.trained_data_fetcher import TrainedDataFetcher
from autoflow.ensemble.trials_fetcher import TrialsFetcher
from autoflow.evaluation.budget import get_default_algo2iter, get_default_algo2budget_mode
from autoflow.evaluation.train_evaluator import TrainEvaluator
from autoflow.hdl.hdl2shps import HDL2SHPS
from autoflow.hdl.hdl_constructor import HDL_Constructor
from autoflow.opt.result_logger import DatabaseResultLogger
from autoflow.opt.utils import get_max_SH_iter, get_budgets
from autoflow.metrics import r2, accuracy
from autoflow.optimizer import Optimizer
from autoflow.resource_manager.base import ResourceManager
from autoflow.utils.config_space import replace_phps
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
            min_budget: Optional[float] = None,
            max_budget: Optional[float] = None,
            eta: Optional[float] = None,
            SH_only: bool = False,
            budget2kfold: Optional[Dict[float, int]] = None,
            algo2budget_mode: Optional[Dict[str, str]] = None,
            algo2iter: Optional[Dict[str, int]] = None,
            only_use_subsamples_budget_mode: bool = False,
            n_folds: int = 5,
            holdout_test_size: float = 1 / 3,
            n_keep_samples: int = 30000,
            min_n_samples_for_SH: int = 1000,
            max_n_samples_for_CV: int = 5000,
            warm_start=True,
            config_generator: Union[str, Type] = "ET",
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
            highR_cat_threshold: float = 0.3,
            consider_ordinal_as_cat: bool = False,
            should_store_intermediate_result: bool = False,
            should_finally_fit: bool = False,
            should_calc_all_metrics: bool = True,
            should_stack_X: bool = True,
            debug_evaluator: bool = False,
            **kwargs
    ):
        '''
        Parameters
        ----------

        hdl_constructor: :class:`autoflow.hdl.hdl_constructor.HDL_Constructor` or None
            ``HDL`` is abbreviation of Hyper-parameter Descriptions Language.

            It describes an abstract hyperparametric space that independent with concrete implementation.

            ``HDL_Constructor`` is a class who is responsible for translating dict-type ``DAG-workflow`` into ``H.D.L`` .

        resource_manager: :class:`autoflow.manager.resource_manager.ResourceManager` or None
            ``ResourceManager`` is a class manager computer resources such like ``file_system`` and ``data_base``.

        random_state: int
            random state

        log_path: path
            which file to store log, if is None, ``autoflow.log`` will be used.

        log_config: dict
            logging configuration

        highR_nan_threshold: float
            high ratio NaN threshold, you can find example and practice in :class:`autoflow.hdl.hdl_constructor.HDL_Constructor`

        highR_cat_threshold: float
            high ratio categorical feature's cardinality threshold, you can find example and practice in :class:`autoflow.hdl.hdl_constructor.HDL_Constructor`

        kwargs
            if parameters like  or ``hdl_constructor`` and ``resource_manager`` are passing None,

            you can passing kwargs to make passed parameter work. See the following example.

        Examples
        ---------
        In this example, you can see a trick to seed kwargs parameters with out initializing
        :class:`autoflow.hdl.hdl_constructor.HDL_Constructor` or other class.

        In following example, user pass ``DAG_workflow`` and ``hdl_bank`` by key-work arguments method.
        And we can see  hdl_constructor is instanced by kwargs implicitly.

        >>> from autoflow import AutoFlowClassifier
        >>> classifier = AutoFlowClassifier(DAG_workflow={"num->target":["lightgbm"]},
        ...   hdl_bank={"classification":{"lightgbm":{"boosting_type":  {"_type": "choice", "_value":["gbdt","dart","goss"]}}}})
        AutoFlowClassifier(hdl_constructor=HDL_Constructor(
            DAG_workflow={'num->target': ['lightgbm']}
            hdl_bank_path=None
            hdl_bank={'classification': {'lightgbm': {'boosting_type': {'_type': 'choice', '_value': ['gbdt', 'dart', 'goss']}}}}
            included_classifiers=('adaboost', 'catboost', 'decision_tree', 'extra_trees', 'gaussian_nb', 'k_nearest_neighbors', 'liblinear_svc', 'lib...
        '''
        self.warm_start = warm_start
        self.debug_evaluator = debug_evaluator
        self.only_use_subsamples_budget_mode = only_use_subsamples_budget_mode
        if algo2iter is None:
            algo2iter = get_default_algo2iter()
        if algo2budget_mode is None:
            algo2budget_mode = get_default_algo2budget_mode()
        if only_use_subsamples_budget_mode:
            algo2budget_mode = {key: SUBSAMPLES_BUDGET_MODE for key in algo2budget_mode}
        self.algo2iter = algo2iter
        self.algo2budget_mode = algo2budget_mode
        self.budget2kfold = budget2kfold
        self.max_n_samples_for_CV = max_n_samples_for_CV
        self.min_n_samples_for_SH = min_n_samples_for_SH
        self.n_keep_samples = n_keep_samples
        self.config_generator_params = dict(config_generator_params)
        assert isinstance(n_folds, int) and n_folds >= 1
        # fixme: support int
        assert isinstance(holdout_test_size, float) and 0 < holdout_test_size < 1
        self.holdout_test_size = holdout_test_size
        self.n_folds = n_folds
        self.min_n_workers = min_n_workers
        self.master_host = master_host
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
        self.should_finally_fit = should_finally_fit
        self.should_store_intermediate_result = should_store_intermediate_result
        self.should_calc_all_metrics = should_calc_all_metrics
        self.log_config = log_config
        self.highR_nan_threshold = highR_nan_threshold
        self.highR_cat_threshold = highR_cat_threshold
        # ---logger------------------------------------
        self.log_path = os.path.expandvars(os.path.expanduser(log_path))
        setup_logger(self.log_path, self.log_config)
        self.logger = get_logger(self)
        # ---random_state-----------------------------------
        self.random_state = random_state
        # ---hdl_constructor--------------------------
        self.hdl_constructor = instancing(hdl_constructor, HDL_Constructor, kwargs)
        # ---resource_manager-----------------------------------
        self.resource_manager: ResourceManager = instancing(resource_manager, ResourceManager, kwargs)
        # ---member_variable------------------------------------
        self.estimator = None
        self.ensemble_estimator = None
        self.evaluators = []
        self.data_manager = None
        self.NS = None
        self.optimizer = None

    def hdl2configSpce(self, hdl: Dict):
        hdl2shps = HDL2SHPS()
        hdl2shps.set_task(self.ml_task)
        return hdl2shps(hdl)

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
            self.highR_cat_threshold, self.consider_ordinal_as_cat, upload_type
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
        # parse splitter
        self.groups = groups
        self.n_samples = self.data_manager.X_train.shape[0]
        if splitter is None:
            if self.n_samples > self.max_n_samples_for_CV:
                self.logger.info(
                    f"TrainSet has {self.n_samples} samples,"
                    f" greater than max n_samples for Cross-Validation "
                    f"(max_n_samples_for_CV = {self.max_n_samples_for_CV}).")
                splitter = ShuffleSplit(n_splits=1, test_size=self.holdout_test_size, random_state=self.random_state)
            else:
                if self.n_folds == 1:
                    splitter = ShuffleSplit(n_splits=1, test_size=self.holdout_test_size,
                                            random_state=self.random_state)
                else:
                    if self.ml_task.mainTask == "classification":
                        splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                    else:
                        splitter = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        assert hasattr(splitter, "split") and hasattr(splitter, "n_splits"), \
            "Parameter 'splitter' should be a train-valid splitter, " \
            "which contain 'split(X, y, groups)' method, 'n_splits' attribute."
        if getattr(splitter, "random_state") is None:
            self.logger.warning(
                f"splitter '{splitter}' haven't specific random_state, it's random_state is set default '{self.random_state}'")
            splitter.random_state = self.random_state
        self.n_splits = splitter.n_splits
        self.splitter = splitter
        # do subsample if n_samples >  n_keep_samples
        if self.n_samples > self.n_keep_samples:
            self.logger.info(
                f"TrainSet has {self.n_samples} samples,"
                f" greater than n_keep_samples({self.n_keep_samples}). ")
        # set min_budget, max_budget, eta, budget2kfold
        if self.n_samples < self.min_n_samples_for_SH:
            self.min_budget = self.max_budget = self.eta = 1
        else:
            # todo: 可以改进得更灵活
            if self.min_budget is None:
                self.min_budget = 1 / 16
            if self.max_budget is None:
                if self.n_splits > 1:
                    self.max_budget = 4
                else:
                    self.max_budget = 1
            if self.eta is None:
                self.eta = 4
            if self.budget2kfold is None:
                if self.max_budget <= 1:
                    self.budget2kfold = {}
                else:
                    self.budget2kfold = {self.max_budget: self.n_splits}
        # set n_iterations
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
        self.hdl_constructor.run(self.data_manager, self.model_registry)
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
            "budget2kfold": self.budget2kfold,
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

    def insert_experiment_record(
            self,
            additional_info: dict = frozendict(),
            fit_ensemble_params: Union[str, Dict[str, Any], None, bool] = "auto",

    ):
        additional_info = dict(additional_info)
        # now we get task_id and hdl_id, we can insert current runtime information into "experiments.experiments" database
        experiment_config = {
            "should_stack_X": self.should_stack_X,
            "should_finally_fit": self.should_finally_fit,
            "should_calc_all_metric": self.should_calc_all_metrics,
            "should_store_intermediate_result": self.should_store_intermediate_result,
            "fit_ensemble_params": str(fit_ensemble_params),
            "highR_nan_threshold": self.highR_nan_threshold,
            "highR_cat_threshold": self.highR_cat_threshold,
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
            should_finally_fit=self.should_finally_fit,
            model_registry=self.model_registry,
            budget2kfold=self.budget2kfold,
            algo2budget_mode=self.algo2budget_mode,
            algo2iter=self.algo2iter,
            max_budget=self.max_budget,
            nameserver=self.ns_host,
            nameserver_port=self.ns_port,
            host=worker_host,
            worker_id=worker_id,
            timeout=None,
            debug=self.debug_evaluator,
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

    # Semantic compatibility with opt
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
        if inspect.isclass(self.config_generator):
            self.logger.info(f"'{self.config_generator}' is a class from '{self.config_generator.__module__}' module.")
            cg_cls = self.config_generator
        elif isinstance(self.config_generator, str):
            module = import_module("autoflow.opt.config_generators")
            cg_cls = getattr(module, self.config_generator)
        else:
            raise NotImplementedError
        budgets = get_budgets(self.min_budget, self.max_budget, self.eta)
        config_generator = cg_cls(self.config_space, budgets, self.random_state, **self.config_generator_params)
        self.database_result_logger = DatabaseResultLogger(self.resource_manager)
        if self.warm_start:
            previous_result, incumbents, incumbent_performances = self.resource_manager.get_result_from_trial_table(
                task_id=self.task_id,
                hdl_id=self.hdl_id,
                user_id=self.user_id,
                budget_id=self.budget_id,
            )
        else:
            previous_result, incumbents, incumbent_performances = None, None, None
        self.optimizer = Optimizer(
            self.run_id,
            config_generator,
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
        self.hpbandstr_result = self.optimizer.run(self.n_iterations, self.min_n_workers)

    # Semantic compatibility with opt
    run_master = run_optimizer

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
        self.run_nameserver()
        self.run_evaluators()
        self.run_optimizer()
        self.optimizer.shutdown(shutdown_workers=True)
        self.NS.shutdown()
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
            should_finally_fit=self.should_finally_fit,
            model_registry=self.model_registry,
            instance_id=self.instance_id
        )



    def fit_ensemble(
            self,
            task_id=None,
            hdl_id=None,
            budget_id=None,
            trials_fetcher="GetBestK",
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
        self.hdl_id = hdl_id
        self.budget_id = budget_id
        self.resource_manager.task_id = task_id
        self.resource_manager.hdl_id = hdl_id
        self.resource_manager.budget_id = budget_id
        if fit_ensemble_alone:
            setup_logger(self.log_path, self.log_config)
            if fit_ensemble_alone:
                experiment_config = {
                    "fit_ensemble_params": kwargs
                }
                self.resource_manager.insert_experiment_record(ExperimentType.ENSEMBLE, experiment_config, {})
                self.experiment_id = self.resource_manager.experiment_id
        trials_fetcher_name = trials_fetcher
        from autoflow.ensemble import trials_fetcher
        assert hasattr(trials_fetcher, trials_fetcher_name)
        trials_fetcher_cls = getattr(trials_fetcher, trials_fetcher_name)
        trials_fetcher: TrialsFetcher = trials_fetcher_cls(
            resource_manager=self.resource_manager,
            task_id=task_id,
            hdl_id=hdl_id,
            **trials_fetcher_params
        )
        trial_ids = trials_fetcher.fetch()
        estimator_list, y_true_indexes_list, y_preds_list = TrainedDataFetcher(
            task_id, hdl_id, trial_ids, self.resource_manager).fetch()
        # todo: 在这里，只取了验证集的数据，没有取测试集的数据。待拓展
        ml_task, y_true = self.resource_manager.get_ensemble_needed_info(task_id)
        ensemble_estimator_package_name = f"autoflow.ensemble.{ensemble_type}.{ml_task.role}"
        ensemble_estimator_package = import_module(ensemble_estimator_package_name)
        ensemble_estimator_class_name = get_class_name_of_module(ensemble_estimator_package_name)
        ensemble_estimator_class = getattr(ensemble_estimator_package, ensemble_estimator_class_name)
        ensemble_estimator: EnsembleEstimator = ensemble_estimator_class(**ensemble_params)
        ensemble_estimator.fit_trained_data(estimator_list, y_true_indexes_list, y_preds_list, y_true)
        self.ensemble_estimator = ensemble_estimator
        if fit_ensemble_alone:
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
        res = deepcopy(self)
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

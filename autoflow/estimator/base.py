import datetime
import math
import multiprocessing
import os
from copy import deepcopy
from importlib import import_module
from multiprocessing import Manager
from typing import Union, Optional, Dict, List, Any

import numpy as np
import pandas as pd
from frozendict import frozendict
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from autoflow import constants
from autoflow.ensemble.base import EnsembleEstimator
from autoflow.ensemble.trained_data_fetcher import TrainedDataFetcher
from autoflow.ensemble.trials_fetcher import TrialsFetcher
from autoflow.hdl.hdl_constructor import HDL_Constructor
from autoflow.manager.data_manager import DataManager
from autoflow.manager.resource_manager import ResourceManager
from autoflow.metrics import r2, accuracy
from autoflow.pipeline.dataframe import GenericDataFrame
from autoflow.tuner.tuner import Tuner
from autoflow.utils.concurrence import get_chunks
from autoflow.utils.config_space import replace_phps, estimate_config_space_numbers
from autoflow.utils.dict import update_mask_from_other_dict
from autoflow.utils.klass import instancing, sequencing
from autoflow.utils.logging_ import get_logger, setup_logger
from autoflow.utils.packages import get_class_name_of_module


class AutoFlowEstimator(BaseEstimator):
    checked_mainTask = None

    def __init__(
            self,
            tuner: Union[Tuner, List[Tuner], None, dict] = None,
            hdl_constructor: Union[HDL_Constructor, List[HDL_Constructor], None, dict] = None,
            resource_manager: Union[ResourceManager, str] = None,
            random_state=42,
            log_file: str = None,
            log_config: Optional[dict] = None,
            highR_nan_threshold=0.5,
            highR_cat_threshold=0.5,
            should_store_intermediate_result=False,
            should_finally_fit=False,
            should_calc_all_metrics=True,
            **kwargs
    ):
        '''
        Parameters
        ----------
        tuner: :class:`autoflow.tuner.tuner.Tuner` or None
            ``Tuner`` if class who agent an abstract search process.

        hdl_constructor: :class:`autoflow.hdl.hdl_constructor.HDL_Constructor` or None
            ``HDL`` is abbreviation of Hyper-parameter Descriptions Language.

            It describes an abstract hyperparametric space that independent with concrete implementation.

            ``HDL_Constructor`` is a class who is responsible for translating dict-type ``DAG-workflow`` into ``H.D.L`` .

        resource_manager: :class:`autoflow.manager.resource_manager.ResourceManager` or None
            ``ResourceManager`` is a class manager computer resources such like ``file_system`` and ``data_base``.

        random_state: int
            random state

        log_file: path
            which file to store log, if is None, ``autoflow.log`` will be used.

        log_config: dict
            logging configuration

        highR_nan_threshold: float
            high ratio NaN threshold, you can find example and practice in :class:`autoflow.hdl.hdl_constructor.HDL_Constructor`

        highR_cat_threshold: float
            high ratio categorical feature's cardinality threshold, you can find example and practice in :class:`autoflow.hdl.hdl_constructor.HDL_Constructor`

        kwargs
            if parameters like ``tuner`` or ``hdl_constructor`` and ``resource_manager`` are passing None,

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
        self.should_finally_fit = should_finally_fit
        self.should_store_intermediate_result = should_store_intermediate_result
        self.should_calc_all_metrics = should_calc_all_metrics
        self.log_config = log_config
        self.highR_nan_threshold = highR_nan_threshold
        self.highR_cat_threshold = highR_cat_threshold

        # ---logger------------------------------------
        self.log_file = log_file
        setup_logger(self.log_file, self.log_config)
        self.logger = get_logger(self)
        # ---random_state-----------------------------------
        self.random_state = random_state
        # ---tuner-----------------------------------
        tuner = instancing(tuner, Tuner, kwargs)
        # ---tuners-----------------------------------
        self.tuners = sequencing(tuner, Tuner)
        self.tuner = self.tuners[0]
        # ---hdl_constructor--------------------------
        hdl_constructor = instancing(hdl_constructor, HDL_Constructor, kwargs)
        # ---hdl_constructors-------------------------
        self.hdl_constructors = sequencing(hdl_constructor, HDL_Constructor)
        self.hdl_constructor = self.hdl_constructors[0]
        # ---resource_manager-----------------------------------
        self.resource_manager = instancing(resource_manager, ResourceManager, kwargs)
        # ---member_variable------------------------------------
        self.estimator = None
        self.ensemble_estimator = None

    def smbo_transfer_learn(self, transfer_tasks, transfer_hdls):
        # 调用这个函数的时候，task_id 和 hdl_id 都建立了
        new_task_id = self.resource_manager.task_id
        if transfer_tasks is not None:
            for old_task_id in transfer_tasks:
                # hdl_id使用当前的hdl_id
                hdl_id = self.resource_manager.hdl_id
                # 根据old_task_id + hdl_id就能定位到runhistory数据表，把数据取出来存到当前的runhistory数据表
                self.resource_manager.migrate_runhistory(old_task_id, hdl_id, new_task_id, hdl_id)

    def fit(
            self,
            X_train: Union[np.ndarray, pd.DataFrame, GenericDataFrame],
            y_train=None,
            X_test=None,
            y_test=None,
            column_descriptions: Optional[Dict] = None,
            metric=None,
            splitter=KFold(5, True, 42),
            specific_task_token="",
            transfer_tasks: Union[List[str], str, None] = None,
            transfer_hdls: Union[List[str], str, None] = None,
            additional_info: dict = frozendict(),
            dataset_metadata: dict = frozenset(),
            task_metadata: dict = frozendict(),
            fit_ensemble_params: Union[str, Dict[str, Any], None, bool] = "auto",

    ):
        '''

        Parameters
        ----------
        X_train: :class:`numpy.ndarray` or :class:`pandas.DataFrame`
        y_train: :class:`numpy.ndarray` or str
        X_test: :class:`numpy.ndarray` or :class:`pandas.DataFrame` or None
        y_test: :class:`numpy.ndarray` or str or None
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
        dataset_metadata = dict(dataset_metadata)
        additional_info = dict(additional_info)
        task_metadata = dict(task_metadata)
        if isinstance(transfer_tasks, str):
            transfer_tasks = [transfer_tasks]
        if isinstance(transfer_hdls, str):
            transfer_hdls = [transfer_hdls]
        # build data_manager
        self.data_manager = DataManager(
            X_train, y_train, X_test, y_test, dataset_metadata, column_descriptions, self.highR_nan_threshold
        )
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
        self.resource_manager.insert_to_tasks_table(self.data_manager, metric, splitter,
                                                    specific_task_token, dataset_metadata, task_metadata)
        self.resource_manager.close_tasks_table()
        # store other params
        self.splitter = splitter
        assert len(self.hdl_constructors) == len(self.tuners)
        n_step = len(self.hdl_constructors)
        general_experiment_timestamp = datetime.datetime.now()
        for step, (hdl_constructor, tuner) in enumerate(zip(self.hdl_constructors, self.tuners)):
            current_experiment_timestamp = datetime.datetime.now()
            hdl_constructor.run(self.data_manager, self.random_state, self.highR_cat_threshold)
            raw_hdl = hdl_constructor.get_hdl()
            if step != 0:
                last_best_dhp = self.resource_manager.load_best_dhp()
                hdl = update_mask_from_other_dict(raw_hdl, last_best_dhp)
                self.logger.debug(f"Updated HDL(Hyperparams Descriptions Language) in step {step}:\n{hdl}")
            else:
                hdl = raw_hdl
            # get hdl_id, and insert record into "{task_id}.hdls" database
            self.resource_manager.insert_to_hdls_table(hdl, hdl_constructor.hdl_metadata)
            self.resource_manager.close_hdls_table()
            # prepare for transfer learn. load runhistory record from old task database to new database
            self.smbo_transfer_learn(transfer_tasks, transfer_hdls)
            # now we get task_id and hdl_id, we can insert current runtime information into "experiments.experiments" database
            self.resource_manager.insert_to_experiments_table(general_experiment_timestamp,
                                                              current_experiment_timestamp,
                                                              self.hdl_constructors, hdl_constructor, raw_hdl, hdl,
                                                              self.tuners, tuner, self.should_calc_all_metrics,
                                                              self.data_manager,
                                                              column_descriptions,
                                                              dataset_metadata, metric, splitter,
                                                              self.should_store_intermediate_result,
                                                              fit_ensemble_params,
                                                              additional_info, self.should_finally_fit)
            self.resource_manager.close_experiments_table()
            self.task_id = self.resource_manager.task_id
            self.hdl_id = self.resource_manager.hdl_id
            self.experiment_id = self.resource_manager.experiment_id
            self.logger.info(f"task_id:\t{self.task_id}")
            self.logger.info(f"hdl_id:\t{self.hdl_id}")
            self.logger.info(f"experiment_id:\t{self.experiment_id}")
            result = self.start_tuner(tuner, hdl)
            if result["is_manual"] == True:
                break
            if step == n_step - 1:
                self.start_final_step(fit_ensemble_params)

        return self

    def get_sync_dict(self, n_jobs, tuner):
        if n_jobs > 1 and tuner.search_method != "grid":
            sync_dict = Manager().dict()
            sync_dict["exit_processes"] = tuner.exit_processes
        else:
            sync_dict = None
        return sync_dict

    def start_tuner(self, tuner: Tuner, hdl: dict):
        self.logger.debug(f"Start fine tune task, \nwhich HDL(Hyperparams Descriptions Language) is:\n{hdl}")
        self.logger.debug(f"which Tuner is:\n{tuner}")
        tuner.set_data_manager(self.data_manager)
        tuner.set_random_state(self.random_state)
        tuner.set_hdl(hdl)  # just for get shps of tuner
        if estimate_config_space_numbers(tuner.shps) == 1:
            self.logger.info("HDL(Hyperparams Descriptions Language) is a constant space, using manual modeling.")
            dhp, self.estimator = tuner.evaluator.shp2model(tuner.shps.sample_configuration())
            self.estimator.fit(self.data_manager.X_train, self.data_manager.y_train)
            return {"is_manual": True}
        n_jobs = tuner.n_jobs
        run_limits = [math.ceil(tuner.run_limit / n_jobs)] * n_jobs
        is_master_list = [False] * n_jobs
        is_master_list[0] = True
        initial_configs_list = get_chunks(
            tuner.design_initial_configs(n_jobs),
            n_jobs)
        random_states = np.arange(n_jobs) + self.random_state
        sync_dict = self.get_sync_dict(n_jobs, tuner)
        self.resource_manager.clear_pid_list()
        self.resource_manager.close_all()
        resource_managers = [deepcopy(self.resource_manager) for i in range(n_jobs)]
        tuners = [deepcopy(tuner) for i in range(n_jobs)]
        processes = []
        # todo: 重构 sync_dict
        for tuner, resource_manager, run_limit, initial_configs, is_master, random_state in \
                zip(tuners, resource_managers, run_limits, initial_configs_list, is_master_list, random_states):
            args = (tuner, resource_manager, run_limit, initial_configs, is_master, random_state, sync_dict)
            if n_jobs == 1:
                self.run(*args)
            else:
                p = multiprocessing.Process(
                    target=self.run,
                    args=args
                )
                processes.append(p)
                p.start()
        for p in processes:
            p.join()

        return {"is_manual": False}

    def start_final_step(self, fit_ensemble_params):
        if isinstance(fit_ensemble_params, str):
            if fit_ensemble_params == "auto":
                self.logger.info(f"'fit_ensemble_params' is 'auto', use default params to fit_ensemble_params.")
                self.estimator = self.fit_ensemble()
            else:
                raise NotImplementedError
        elif isinstance(fit_ensemble_params, bool):
            if fit_ensemble_params:
                self.logger.info(f"'fit_ensemble_params' is True, use default params to fit_ensemble_params.")
                self.estimator = self.fit_ensemble()
            else:
                self.logger.info(
                    f"'fit_ensemble_params' is False, don't fit_ensemble but use best trial as result.")
                self.estimator = self.resource_manager.load_best_estimator(self.ml_task)
        elif isinstance(fit_ensemble_params, dict):
            self.logger.info(
                f"'fit_ensemble_params' is specific: {fit_ensemble_params}.")
            self.estimator = self.fit_ensemble(**fit_ensemble_params)
        elif fit_ensemble_params is None:
            self.logger.info(
                f"'fit_ensemble_params' is None, don't fit_ensemble but use best trial as result.")
            self.estimator = self.resource_manager.load_best_estimator(self.ml_task)
        else:
            raise NotImplementedError

    def run(self, tuner, resource_manager, run_limit, initial_configs, is_master, random_state, sync_dict=None):
        if sync_dict:
            sync_dict[os.getpid()] = 0
            resource_manager.sync_dict = sync_dict
        resource_manager.set_is_master(is_master)
        resource_manager.push_pid_list()
        # random_state: 1. set_hdl中传给phps 2. 传给所有配置
        tuner.random_state = random_state
        tuner.run_limit = run_limit
        tuner.set_resource_manager(resource_manager)
        # 替换搜索空间中的 random_state
        replace_phps(tuner.shps, "random_state", int(random_state))
        tuner.shps.seed(random_state)
        # todo : 增加 n_jobs ? 调研默认值
        tuner.run(
            initial_configs=initial_configs,
            evaluator_params=dict(
                random_state=random_state,
                data_manager=self.data_manager,
                metric=self.metric,
                should_calc_all_metric=self.should_calc_all_metrics,
                splitter=self.splitter,
                should_store_intermediate_result=self.should_store_intermediate_result,
                resource_manager=resource_manager,
                should_finally_fit=self.should_finally_fit
            ),
            instance_id=resource_manager.task_id,
            rh_db_type=resource_manager.db_type,
            rh_db_params=resource_manager.runhistory_db_params,
            rh_db_table_name=resource_manager.runhistory_table_name
        )
        if sync_dict:
            sync_dict[os.getpid()] = 1

    def fit_ensemble(
            self,
            task_id=None,
            hdl_id=None,
            trials_fetcher="GetBestK",
            trials_fetcher_params=frozendict(k=10),
            ensemble_type="stack",
            ensemble_params=frozendict(),
            return_Xy_test=False
    ):
        if task_id is None:
            assert hasattr(self.resource_manager, "task_id") and self.resource_manager.task_id is not None
            task_id = self.resource_manager.task_id
        # if hdl_id is None:
        #     assert hasattr(self.resource_manager, "hdl_id") and self.resource_manager.hdl_id is not None
        #     hdl_id = self.resource_manager.hdl_id
        trials_fetcher_name = trials_fetcher
        from autoflow.ensemble import trials_fetcher
        assert hasattr(trials_fetcher, trials_fetcher_name)
        trials_fetcher_cls = getattr(trials_fetcher, trials_fetcher_name)
        trials_fetcher: TrialsFetcher = trials_fetcher_cls(resource_manager=self.resource_manager, task_id=task_id,
                                                           hdl_id=hdl_id,
                                                           **trials_fetcher_params)
        trial_ids = trials_fetcher.fetch()
        estimator_list, y_true_indexes_list, y_preds_list = TrainedDataFetcher(
            task_id, hdl_id, trial_ids, self.resource_manager).fetch()
        # todo: 在这里，只取了验证集的数据，没有取测试集的数据。待拓展
        ml_task, Xy_train, Xy_test = self.resource_manager.get_ensemble_needed_info(task_id)
        y_true = Xy_train[1]
        ensemble_estimator_package_name = f"autoflow.ensemble.{ensemble_type}.{ml_task.role}"
        ensemble_estimator_package = import_module(ensemble_estimator_package_name)
        ensemble_estimator_class_name = get_class_name_of_module(ensemble_estimator_package_name)
        ensemble_estimator_class = getattr(ensemble_estimator_package, ensemble_estimator_class_name)
        ensemble_estimator: EnsembleEstimator = ensemble_estimator_class(**ensemble_params)
        # todo: 集成学习部分，存在无法处理K折验证以外问题的情况（不完整的y_pred）
        ensemble_estimator.fit_trained_data(estimator_list, y_true_indexes_list, y_preds_list, y_true)
        self.ensemble_estimator = ensemble_estimator
        if return_Xy_test:
            return self.ensemble_estimator, Xy_test
        else:
            return self.ensemble_estimator

    def auto_fit_ensemble(self):
        # todo: 调研stacking等ensemble方法的表现评估
        pass

    def _predict(
            self,
            X_test,
            task_id=None,
            trial_id=None,
            experiment_id=None,
            column_descriptions: Optional[Dict] = None,
            highR_nan_threshold=0.5
    ):
        is_set_X_test = False
        if hasattr(self, "data_manager") and self.data_manager is not None:
            self.logger.info(
                "'data_manager' is existing in AutoFlowEstimator, will not load it from database or create it.")
        else:
            if task_id is not None:
                _experiment_id = self.resource_manager.get_experiment_id_by_task_id(task_id)
            elif experiment_id is not None:
                _experiment_id = experiment_id
            elif hasattr(self, "experiment_id") and self.experiment_id is not None:
                _experiment_id = self.experiment_id
            else:
                _experiment_id = None
            if _experiment_id is None:
                self.logger.info(
                    "'_experiment_id' is not exist, initializing data_manager by user given parameters.")
                self.data_manager = DataManager(X_test, column_descriptions=column_descriptions,
                                                highR_nan_threshold=highR_nan_threshold)
                is_set_X_test = True
            else:
                self.logger.info(
                    "'_experiment_id' is exist, loading data_manager by query meta_record.experiments database.")
                self.data_manager: DataManager = self.resource_manager.load_data_manager_by_experiment_id(
                    _experiment_id)
        if not is_set_X_test:
            self.data_manager.set_data(X_test=X_test)
        if self.estimator is None:
            self.logger.warning(
                f"'{self.__class__.__name__}' 's estimator is None, maybe you didn't use fit method to train the data.\n"
                f"We try to query trials database if you seed trial_id specifically.")
            raise NotImplementedError

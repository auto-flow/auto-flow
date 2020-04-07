import datetime
import math
import os
from copy import deepcopy
from multiprocessing import Manager
from typing import Union, Optional, Dict, List

import joblib
import json5 as json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from hyperflow.ensemble.stack.builder import StackEnsembleBuilder
from hyperflow.hdl.hdl_constructor import HDL_Constructor
from hyperflow.manager.resource_manager import ResourceManager
from hyperflow.manager.xy_data_manager import XYDataManager
from hyperflow.metrics import r2, accuracy
from hyperflow.pipeline.dataframe import GenericDataFrame
from hyperflow.tuner.tuner import Tuner
from hyperflow.utils.concurrence import get_chunks
from hyperflow.utils.config_space import replace_phps, estimate_config_space_numbers
from hyperflow.utils.dict import update_placeholder_from_other_dict
from hyperflow.utils.logging_ import get_logger


class HyperFlowEstimator(BaseEstimator):

    def __init__(
            self,
            tuner: Union[Tuner, List[Tuner], None, dict] = None,
            hdl_constructor: Union[HDL_Constructor, List[HDL_Constructor], None, dict] = None,
            resource_manager: Union[ResourceManager, str] = None,
            ensemble_builder: Union[StackEnsembleBuilder, None, bool, int] = None,
            random_state=42
    ):
        # ---logger------------------------------------
        self.logger=get_logger(__name__)
        # ---random_state-----------------------------------
        self.random_state = random_state
        # ---ensemble_builder-----------------------------------
        if ensemble_builder is None:
            self.logger.info("Using default StackEnsembleBuilder.")
            ensemble_builder = StackEnsembleBuilder()
        elif ensemble_builder == False:
            self.logger.info("Not using EnsembleBuilder, will select the best estimator.")
        else:
            ensemble_builder = StackEnsembleBuilder(set_model=ensemble_builder)
        self.ensemble_builder = ensemble_builder
        # ---tuners-----------------------------------
        if not tuner:
            tuner = Tuner()
        if not isinstance(tuner, (list, tuple)):
            tuner = [tuner]
        self.tuners: List[Tuner] = tuner
        # ---hdl_constructors-----------------------------------
        if not hdl_constructor:
            hdl_constructor = HDL_Constructor()
        if not isinstance(hdl_constructor, (list, tuple)):
            hdl_constructor = [hdl_constructor]
        self.hdl_constructors = hdl_constructor
        # ---resource_manager-----------------------------------
        if resource_manager is None:
            resource_manager = ResourceManager()
        self.resource_manager = resource_manager
        # ---member_variable------------------------------------
        self.estimator = None


    def fit(
            self,
            X: Union[np.ndarray, pd.DataFrame, GenericDataFrame],
            y=None,
            X_test=None,
            y_test=None,
            column_descriptions: Optional[Dict] = None,
            dataset_metadata=frozenset(),
            metric=None,
            all_scoring_functions=True,
            splitter=KFold(5, True, 42),
    ):
        dataset_metadata = dict(dataset_metadata)
        # build data_manager
        self.data_manager = XYDataManager(
            X, y, X_test, y_test, dataset_metadata, column_descriptions
        )
        self.ml_task = self.data_manager.ml_task
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
        self.resource_manager.insert_to_tasks_db(self.data_manager, metric, splitter)
        self.resource_manager.close_tasks_db()
        # store other params
        self.all_scoring_functions = all_scoring_functions
        self.splitter = splitter
        assert len(self.hdl_constructors) == len(self.tuners)
        n_step = len(self.hdl_constructors)
        general_experiment_timestamp = datetime.datetime.now()
        for step, (hdl_constructor, tuner) in enumerate(zip(self.hdl_constructors, self.tuners)):
            current_experiment_timestamp = datetime.datetime.now()
            hdl_constructor.set_data_manager(self.data_manager)
            hdl_constructor.set_random_state(self.random_state)
            hdl_constructor.run()
            raw_hdl = hdl_constructor.get_hdl()
            if step != 0:
                last_best_dhp = self.resource_manager.load_best_dhp()
                last_best_dhp = json.loads(last_best_dhp)
                hdl = update_placeholder_from_other_dict(raw_hdl, last_best_dhp)
                self.logger.info(f"Updated HDL(Hyperparams Descriptions Language) in step {step}:\n{hdl}")
            else:
                hdl = raw_hdl
            # get hdl_id, and insert record into "{task_id}.hdls" database
            self.resource_manager.insert_to_hdls_db(hdl)
            self.resource_manager.close_hdls_db()
            # now we get task_id and hdl_id, we can insert current runtime information into "experiments.experiments" database
            self.resource_manager.insert_to_experiments_db(general_experiment_timestamp, current_experiment_timestamp,
                                                           self.hdl_constructors, hdl_constructor, raw_hdl, hdl,
                                                           self.tuners, tuner, all_scoring_functions, self.data_manager,
                                                           column_descriptions, dataset_metadata, metric, splitter)
            self.resource_manager.close_experiments_db()

            result = self.start_tuner(tuner, hdl)
            if result["is_manual"] == True:
                break

            if step == n_step - 1:
                if self.ensemble_builder:
                    self.estimator = self.fit_ensemble()
                else:
                    self.estimator = self.resource_manager.load_best_estimator(self.ml_task)
        return self

    def start_tuner(self, tuner: Tuner, hdl: dict):
        self.logger.info(f"Start fine tune task, \nwhich HDL(Hyperparams Descriptions Language) is:\n{hdl}")
        self.logger.info(f"which Tuner is:\n{tuner}")
        tuner.set_data_manager(self.data_manager)
        tuner.set_random_state(self.random_state)
        tuner.set_hdl(hdl)  # just for get shps of tuner
        if estimate_config_space_numbers(tuner.shps) == 1:
            self.logger.info("HDL(Hyperparams Descriptions Language) is a constant space, using manual modeling.")
            dhp, self.estimator = tuner.shp2model(tuner.shps.sample_configuration())
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
        if n_jobs > 1 and tuner.search_method != "grid":
            sync_dict = Manager().dict()
            sync_dict["exit_processes"] = tuner.exit_processes
        else:
            sync_dict = None
        self.resource_manager.close_trials_db()
        self.resource_manager.clear_pid_list()
        self.resource_manager.close_redis()
        resource_managers = [deepcopy(self.resource_manager) for i in range(n_jobs)]
        tuners = [deepcopy(tuner) for i in range(n_jobs)]
        with joblib.parallel_backend(n_jobs=n_jobs, backend="multiprocessing"):
            joblib.Parallel()(
                joblib.delayed(self.run)
                (tuner, resource_manager, run_limit, initial_configs, is_master, random_state, sync_dict)
                for tuner, resource_manager, run_limit, initial_configs, is_master, random_state in
                zip(tuners, resource_managers, run_limits, initial_configs_list, is_master_list, random_states)
            )
        return {"is_manual": False}

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
        replace_phps(tuner.shps, "random_state", int(random_state))
        tuner.shps.seed(random_state)
        tuner.set_addition_info({})
        # todo : 增加 n_jobs ? 调研默认值
        tuner.run(
            self.data_manager,
            self.metric,
            self.all_scoring_functions,
            self.splitter,
            initial_configs
        )
        if sync_dict:
            sync_dict[os.getpid()] = 1

    def fit_ensemble(
            self,
            data_manager=None,
            dataset_paths=None,
    ):
        if not data_manager:
            if hasattr(self, "data_manager"):
                data_manager = self.data_manager
            else:
                if isinstance(dataset_paths, str):
                    dataset_path = dataset_paths
                else:
                    dataset_path = dataset_paths[0]
                data_manager = joblib.load(dataset_path + "/data_manager.bz2")
        if not dataset_paths:
            dataset_paths = self.resource_manager.datasets_dir
        self.ensemble_builder.set_data(
            data_manager,
            dataset_paths,
            self.resource_manager
        )
        self.ensemble_builder.init_data()
        self.stack_estimator = self.ensemble_builder.build()
        return self.stack_estimator

    def refit(
            self,
            dataset_name="default_dataset_name",
            metric=None,
            all_scoring_functions=None,
            splitter=None
    ):
        def update_if_not_None(dict_: Dict, value_name, value):
            if value is not None:
                dict_.update({value_name: value})

        # 不再初始化
        self.tuners.initial_runs = 0
        # resource_manager
        self.resource_manager.load_dataset_path(dataset_name)
        # data_manager
        self.data_manager: XYDataManager = self.resource_manager.load_object("data_manager")
        # hdl default_hp
        hdl_info = self.resource_manager.load_hdl()
        self.hdl = hdl_info["hdl"]
        self.default_hp = hdl_info["default_hp"]
        # evaluate info
        evaluate_info: Dict = self.resource_manager.load_object("evaluate_info")
        update_if_not_None(evaluate_info, "metric", metric)
        update_if_not_None(evaluate_info, "all_scoring_functions", all_scoring_functions)
        update_if_not_None(evaluate_info, "splitter", splitter)
        self.set_dict_to_self(evaluate_info)
        self.ml_task = self.data_manager.ml_task
        # other
        self.resource_manager.dump_db_to_csv()
        # fine tune
        self.run()
        if self.ensemble_builder:
            self.estimator = self.fit_ensemble()
        else:
            self.estimator = self.resource_manager.load_best_estimator(self.ml_task)

        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def set_dict_to_self(self, dict_):
        for key, value in dict_.items():
            setattr(self, key, value)

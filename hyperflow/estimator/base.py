import datetime
import inspect
import math
import os
from copy import deepcopy
from importlib import import_module
from multiprocessing import Manager
from typing import Union, Optional, Dict, List, Any, Sequence

import joblib
import json5 as json
import numpy as np
import pandas as pd
from frozendict import frozendict
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from hyperflow.ensemble.base import EnsembleEstimator
from hyperflow.ensemble.trained_data_fetcher import TrainedDataFetcher
from hyperflow.ensemble.trials_fetcher import TrialsFetcher
from hyperflow.hdl.hdl_constructor import HDL_Constructor
from hyperflow.manager.data_manager import DataManager
from hyperflow.manager.resource_manager import ResourceManager
from hyperflow.metrics import r2, accuracy
from hyperflow.pipeline.dataframe import GenericDataFrame
from hyperflow.tuner.tuner import Tuner
from hyperflow.utils.concurrence import get_chunks
from hyperflow.utils.config_space import replace_phps, estimate_config_space_numbers
from hyperflow.utils.dict import update_placeholder_from_other_dict
from hyperflow.utils.logging import get_logger
from hyperflow.utils.packages import get_class_name_of_module


def get_valid_params_in_kwargs(klass, kwargs: Dict[str, Any]):
    validated = {}
    for key, value in kwargs.items():
        if key in inspect.signature(klass.__init__).parameters.keys():
            validated[key] = value
    return validated


def instancing(variable, klass, kwargs):
    if variable is None:
        variable = klass(**get_valid_params_in_kwargs(klass, kwargs))
    elif isinstance(variable, dict):
        variable = klass(**variable)
    elif isinstance(variable, klass):
        pass
    elif isinstance(variable, Sequence):
        for elem in variable:
            assert isinstance(elem, klass)
    else:
        raise NotImplementedError
    return variable


def sequencing(variable, klass):
    if not isinstance(variable, Sequence):
        variable = [variable]
    variables: Sequence[klass] = variable
    return variables


class HyperFlowEstimator(BaseEstimator):
    checked_mainTask = None

    def __init__(
            self,
            tuner: Union[Tuner, List[Tuner], None, dict] = None,
            hdl_constructor: Union[HDL_Constructor, List[HDL_Constructor], None, dict] = None,
            resource_manager: Union[ResourceManager, str] = None,
            random_state=42,
            **kwargs
    ):
        '''
        Base Estimator of HyperFlow.

        Parameters
        ----------
        tuner: dict
            tuner is a object to fine tune the hyper-parameters.
        hdl_constructor: int
            H.D.L. is a abbreviation of Hyperparams Descriptions Language.
            hdl_constructor is a object to build HDL by seeding specific parameters.
        resource_manager: str
            resource_manager is a object to manager the resources, such like database connections and file-systems.
        '''
        # ---logger------------------------------------
        self.logger = get_logger(__name__)
        # ---random_state-----------------------------------
        self.random_state = random_state
        # ---tuner-----------------------------------
        tuner = instancing(tuner, Tuner, kwargs)
        # ---tuners-----------------------------------
        self.tuners = sequencing(tuner, Tuner)
        # ---hdl_constructor--------------------------
        hdl_constructor = instancing(hdl_constructor, HDL_Constructor, kwargs)
        # ---hdl_constructors-------------------------
        self.hdl_constructors = sequencing(hdl_constructor, HDL_Constructor)
        # ---resource_manager-----------------------------------
        self.resource_manager = instancing(resource_manager, ResourceManager, kwargs)
        # ---member_variable------------------------------------
        self.estimator = None
        self.ensemble_estimator = None

    def fit(
            self,
            X: Union[np.ndarray, pd.DataFrame, GenericDataFrame],
            y=None,
            X_test=None,
            y_test=None,
            column_descriptions: Optional[Dict] = None,
            dataset_metadata: dict = frozenset(),
            metric=None,
            should_calc_all_metrics=True,
            splitter=KFold(5, True, 42),
            specific_task_token="",
            should_store_intermediate_result=False,
            additional_info: dict = frozendict(),
            fit_ensemble_params: Union[str, Dict[str, Any], None, bool] = "auto"
    ):

        self.should_store_intermediate_result = should_store_intermediate_result
        dataset_metadata = dict(dataset_metadata)
        additional_info = dict(additional_info)
        # build data_manager
        self.data_manager = DataManager(
            X, y, X_test, y_test, dataset_metadata, column_descriptions
        )
        self.ml_task = self.data_manager.ml_task
        if self.checked_mainTask is not None:
            assert self.checked_mainTask == self.ml_task.mainTask
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
        self.resource_manager.insert_to_tasks_table(self.data_manager, metric, splitter, specific_task_token)
        self.resource_manager.close_tasks_table()
        # store other params
        self.should_calc_all_metrics = should_calc_all_metrics
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
            self.resource_manager.insert_to_hdls_table(hdl)
            self.resource_manager.close_hdls_table()
            # now we get task_id and hdl_id, we can insert current runtime information into "experiments.experiments" database
            self.resource_manager.insert_to_experiments_table(general_experiment_timestamp,
                                                              current_experiment_timestamp,
                                                              self.hdl_constructors, hdl_constructor, raw_hdl, hdl,
                                                              self.tuners, tuner, should_calc_all_metrics,
                                                              self.data_manager,
                                                              column_descriptions,
                                                              dataset_metadata, metric, splitter,
                                                              should_store_intermediate_result, fit_ensemble_params,
                                                              additional_info)
            self.resource_manager.close_experiments_table()

            result = self.start_tuner(tuner, hdl)
            if result["is_manual"] == True:
                break

            if step == n_step - 1:
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
        return self

    def start_tuner(self, tuner: Tuner, hdl: dict):
        self.logger.info(f"Start fine tune task, \nwhich HDL(Hyperparams Descriptions Language) is:\n{hdl}")
        self.logger.info(f"which Tuner is:\n{tuner}")
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
        if n_jobs > 1 and tuner.search_method != "grid":
            sync_dict = Manager().dict()
            sync_dict["exit_processes"] = tuner.exit_processes
        else:
            sync_dict = None
        self.resource_manager.close_trials_table()
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
                resource_manager=resource_manager
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
        if hdl_id is None:
            assert hasattr(self.resource_manager, "hdl_id") and self.resource_manager.hdl_id is not None
            hdl_id = self.resource_manager.hdl_id
        trials_fetcher_name = trials_fetcher
        from hyperflow.ensemble import trials_fetcher
        assert hasattr(trials_fetcher, trials_fetcher_name)
        trials_fetcher_cls = getattr(trials_fetcher, trials_fetcher_name)
        trials_fetcher: TrialsFetcher = trials_fetcher_cls(resource_manager=self.resource_manager, task_id=task_id,
                                                           hdl_id=hdl_id,
                                                           **trials_fetcher_params)
        trial_ids = trials_fetcher.fetch()
        estimator_list, y_true_indexes_list, y_preds_list = TrainedDataFetcher(
            task_id, hdl_id, trial_ids, self.resource_manager).fetch()
        ml_task, Xy_train, Xy_test = self.resource_manager.get_ensemble_needed_info(task_id, hdl_id)
        y_true = Xy_train[1]
        ensemble_estimator_package_name = f"hyperflow.ensemble.{ensemble_type}.{ml_task.role}"
        ensemble_estimator_package = import_module(ensemble_estimator_package_name)
        ensemble_estimator_class_name = get_class_name_of_module(ensemble_estimator_package_name)
        ensemble_estimator_class = getattr(ensemble_estimator_package, ensemble_estimator_class_name)
        ensemble_estimator: EnsembleEstimator = ensemble_estimator_class(**ensemble_params)
        ensemble_estimator.fit_trained_data(estimator_list, y_true_indexes_list, y_preds_list, y_true)
        self.ensemble_estimator = ensemble_estimator
        if return_Xy_test:
            return self.ensemble_estimator, Xy_test
        else:
            return self.ensemble_estimator

    def auto_fit_ensemble(self):
        pass

    def predict(self, X):
        return self.estimator.predict(X)

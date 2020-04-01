import math
import os
from copy import deepcopy
from multiprocessing import Manager
from typing import Union, Optional, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from autopipeline.ensemble.stack.builder import StackEnsembleBuilder
from autopipeline.hdl.hdl_constructor import HDL_Constructor
from autopipeline.manager.resource_manager import ResourceManager
from autopipeline.manager.xy_data_manager import XYDataManager
from autopipeline.metrics import r2, accuracy
from autopipeline.pipeline.dataframe import GenericDataFrame
from autopipeline.tuner.smac_tuner import Tuner
from autopipeline.utils.concurrence import parse_n_jobs
from autopipeline.utils.config_space import replace_phps
from autopipeline.utils.data import get_chunks


class AutoPipelineEstimator(BaseEstimator):

    def __init__(
            self,
            tuner: Union[Tuner, List[Tuner], None, dict] = None,
            hdl_constructor: Union[HDL_Constructor, List[HDL_Constructor], None, dict] = None,
            resource_manager: Union[ResourceManager, str] = None,
            ensemble_builder: Union[StackEnsembleBuilder, None, bool, int] = None,
            random_state=42
    ):
        self.random_state = random_state
        if ensemble_builder is None:
            print("info: 使用默认的stack集成学习器")
            ensemble_builder = StackEnsembleBuilder()
        elif ensemble_builder == False:
            print("info: 不使用集成学习")
        else:
            ensemble_builder = StackEnsembleBuilder(set_model=ensemble_builder)
        self.ensemble_builder = ensemble_builder
        # todo: 将tuner的参数提到上面来
        if not tuner:
            tuner = Tuner()
        # if not isinstance(tuner,(list,tuple)):
        #     tuner=[tuner]
        self.tuner: List[Tuner] = tuner
        if not hdl_constructor:
            hdl_constructor = HDL_Constructor()
        if isinstance(hdl_constructor, dict):
            # todo: 使用用户自定义超参描述语言
            print("使用用户自定义超参描述语言")
        self.hdl_constructor = hdl_constructor
        self.random_state = tuner.random_state
        if isinstance(resource_manager, str):
            resource_manager = ResourceManager(resource_manager)  # todo : 识别不同协议的文件系统，例如hdfs
        elif resource_manager is None:
            resource_manager = ResourceManager()
        self.resource_manager = resource_manager
        self.estimator = None

    def fit(
            self,
            X: Union[np.ndarray, pd.DataFrame, GenericDataFrame],
            y=None,
            X_test=None,
            y_test=None,
            column_descriptions: Optional[Dict] = None,
            dataset_name="default_dataset_name",
            metric=None,
            all_scoring_functions=False,
            splitter=KFold(5, True, 42),
            n_jobs=1,
            exit_processes=None
    ):
        self.n_jobs = parse_n_jobs(n_jobs)
        if exit_processes is None:
            exit_processes = max(self.n_jobs // 3, 1)
        self.exit_processes = exit_processes
        # resource_manager
        self.resource_manager.init_dataset_path(dataset_name)
        # data_manager
        self.data_manager = XYDataManager(
            X, y, X_test, y_test, dataset_name, column_descriptions
        )
        self.resource_manager.dump_object("data_manager", self.data_manager)
        # hdl default_hp
        self.hdl_constructor.set_data_manager(self.data_manager)
        self.hdl_constructor.set_random_state(self.tuner.random_state)
        self.hdl_constructor.run()
        self.resource_manager.dump_hdl(self.hdl_constructor)
        self.hdl = self.hdl_constructor.get_hdl()
        # fixme
        self.default_hp = {}  # self.hdl_constructor.get_default_hp()
        # evaluate_info
        self.task = self.data_manager.task
        if metric is None:
            if self.task.mainTask == "regression":
                metric = r2
            elif self.task.mainTask == "classification":
                metric = accuracy
            else:
                raise NotImplementedError()
        self.metric = metric
        self.all_scoring_functions = all_scoring_functions
        self.splitter = splitter
        self.evaluate_info = {
            "metric": self.metric,
            "all_scoring_functions": self.all_scoring_functions,
            "splitter": self.splitter,
        }
        self.resource_manager.dump_object("evaluate_info", self.evaluate_info)
        # fine tune
        self.tuner.set_task(self.data_manager.task)
        self.tuner.set_random_state(self.random_state)
        self.start_tuner()
        if self.ensemble_builder:
            self.estimator = self.fit_ensemble()
        else:
            self.estimator = self.resource_manager.load_best_estimator(self.task)
        return self

    def start_tuner(self):
        self.tuner.set_hdl(self.hdl)  # just for get phps of tunner
        n_jobs = self.n_jobs
        run_limits = [math.ceil(self.tuner.run_limit / n_jobs)] * n_jobs
        is_master_list = [False] * n_jobs
        is_master_list[0] = True
        initial_configs_list = get_chunks(
            self.tuner.design_initial_configs(),
            n_jobs)
        random_states = np.arange(n_jobs) + self.random_state
        if n_jobs > 1:
            sync_dict = Manager().dict()
            sync_dict["exit_processes"] = self.exit_processes
        else:
            sync_dict = None
        with joblib.parallel_backend(n_jobs=n_jobs, backend="multiprocessing"):
            joblib.Parallel()(
                joblib.delayed(self.run)
                (run_limit, initial_configs, is_master, random_state, sync_dict)
                for run_limit, initial_configs, is_master, random_state in
                zip(run_limits, initial_configs_list, is_master_list, random_states)
            )

    def run(self, run_limit, initial_configs, is_master, random_state, sync_dict=None):
        tuner = deepcopy(self.tuner)
        resource_manager = deepcopy(self.resource_manager)
        # resource_manager
        if sync_dict:
            sync_dict[os.getpid()] = 0
            resource_manager.sync_dict = sync_dict
        resource_manager.set_is_master(is_master)
        resource_manager.smac_output_dir += (f"/{os.getpid()}")
        # random_state: 1. set_hdl中传给phps 2. 传给所有配置
        tuner.random_state = random_state
        tuner.run_limit = run_limit
        tuner.set_resource_manager(resource_manager)
        tuner.set_data_manager(self.data_manager)
        replace_phps(tuner.phps, "random_state", int(random_state))
        tuner.phps.seed(random_state)
        tuner.set_addition_info({})  # {"shape": X.shape}
        tuner.evaluator.set_resource_manager(resource_manager)
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
            dataset_paths = self.resource_manager.dataset_path
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
        self.tuner.initial_runs = 0
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
        self.task = self.data_manager.task
        # other
        self.resource_manager.dump_db_to_csv()
        # fine tune
        self.run()
        if self.ensemble_builder:
            self.estimator = self.fit_ensemble()
        else:
            self.estimator = self.resource_manager.load_best_estimator(self.task)

        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def set_dict_to_self(self, dict_):
        for key, value in dict_.items():
            setattr(self, key, value)

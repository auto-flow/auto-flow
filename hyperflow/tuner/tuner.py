import random
import re
import time
import os
from typing import Dict, Optional

import numpy as np
from ConfigSpace import ConfigurationSpace
from frozendict import frozendict

from dsmac.facade.smac_hpo_facade import SMAC4HPO
from dsmac.scenario.scenario import Scenario
from hyperflow.constants import MLTask
from hyperflow.evaluation.train_evaluator import TrainEvaluator
from hyperflow.hdl2shps.hdl2shps import HDL2SHPS
from hyperflow.manager.resource_manager import ResourceManager
from hyperflow.manager.data_manager import DataManager
from hyperflow.metrics import Scorer
from hyperflow.pipeline.pipeline import GenericPipeline
from hyperflow.shp2dhp.shp2dhp import SHP2DHP
from hyperflow.utils.concurrence import parse_n_jobs
from hyperflow.utils.config_space import get_random_initial_configs, get_grid_initial_configs
from hyperflow.utils.dict import group_dict_items_before_first_dot
from hyperflow.utils.logging_ import get_logger
from hyperflow.utils.packages import get_class_object_in_pipeline_components
from hyperflow.utils.pipeline import concat_pipeline


class Tuner():
    def __init__(
            self,
            search_method: str = "smac",
            run_limit: int = 100,
            initial_runs: int = 20,
            search_method_params: dict = frozendict(),
            n_jobs: int = 1,
            exit_processes: Optional[int] = None
    ):
        self.search_method_params = search_method_params
        assert search_method in ("smac", "grid", "random")
        if search_method in ("grid", "random"):
            initial_runs = 0
        self.initial_runs = initial_runs
        self.run_limit = run_limit
        self.evaluator = TrainEvaluator()
        self.search_method = search_method
        self.evaluator.set_shp2model(self.shp2model)
        self.random_state = 0
        self.addition_info = {}
        self.resource_manager = None
        self.ml_task = None
        self.data_manager = None
        self.n_jobs = parse_n_jobs(n_jobs)
        if exit_processes is None:
            exit_processes = max(self.n_jobs // 3, 1)
        self.exit_processes = exit_processes
        self.logger=get_logger(__name__)

    def __str__(self):
        return (
            f"hyperflow.Tuner("
            f"search_method={repr(self.search_method)}, "
            f"run_limit={repr(self.run_limit)}, "
            f"initial_runs={repr(self.initial_runs)}, "
            f"search_method_params={repr(self.search_method_params)}, "
            f"n_jobs={repr(self.n_jobs)}, "
            f"exit_processes={repr(self.exit_processes)}"
            f")")

    __repr__ = __str__

    def set_random_state(self, random_state):
        self.random_state = random_state

    def set_addition_info(self, addition_info):
        self.addition_info = addition_info

    def set_hdl(self, hdl: Dict):
        self.hdl = hdl
        # todo: 泛化ML管线后，可能存在多个preprocessing
        self.shps: ConfigurationSpace = self.hdl2phps(hdl)
        self.shps.seed(self.random_state)

    def set_resource_manager(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.evaluator.set_resource_manager(resource_manager)

    def set_task(self, ml_task: MLTask):
        self.ml_task = ml_task

    def set_data_manager(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.ml_task=data_manager.ml_task

    def design_initial_configs(self, n_jobs):
        if self.search_method == "smac":
            return get_random_initial_configs(self.shps, max(self.initial_runs, n_jobs), self.random_state)
        elif self.search_method == "grid":
            return get_grid_initial_configs(self.shps, self.run_limit, self.random_state)
        elif self.search_method == "random":
            return get_random_initial_configs(self.shps, self.run_limit, self.random_state)
        else:
            raise NotImplementedError

    def get_run_limit(self):
        if self.search_method == "smac":
            return self.run_limit
        else:
            return 0

    def run(
            self,
            data_manager: DataManager,
            metric: Scorer,
            all_scoring_functions: bool,
            splitter,
            initial_configs,
            should_store_intermediate_result
    ):
        # time.sleep(random.random())
        if not initial_configs:
            self.logger.warning("Haven't initial_configs. Return.")
            return
        if hasattr(splitter, "random_state"):
            setattr(splitter, "random_state", self.random_state)
        self.evaluator.init_data(
            data_manager,
            metric,
            all_scoring_functions,
            splitter,
            should_store_intermediate_result
        )

        self.scenario = Scenario(
            {
                "run_obj": "quality",
                "runcount-limit": 1000,
                "cs": self.shps,  # configuration space
                "deterministic": "true",
                # todo : 如果是local，存在experiment，如果是其他文件系统，不输出smac
                # "output_dir": self.resource_manager.smac_output_dir,
            },
            initial_runs=0,
            db_type=self.resource_manager.db_type,
            db_params=self.resource_manager.get_runhistory_db_params(),
            anneal_func=self.search_method_params.get("anneal_func")
        )
        # todo 将 file_system 传入，或者给file_system添加 runtime 参数
        smac = SMAC4HPO(
            scenario=self.scenario,
            rng=np.random.RandomState(self.random_state),
            tae_runner=self.evaluator,
            initial_configurations=initial_configs
        )
        smac.solver.initial_configurations = initial_configs
        smac.solver.start_()
        run_limit = self.get_run_limit()
        for i in range(run_limit):
            smac.solver.run_()
            should_continue = self.evaluator.resource_manager.delete_models()
            if not should_continue:
                self.logger.info(f"PID = {os.getpid()} is exiting.")
                break

    def shp2model(self, shp):
        php2dhp = SHP2DHP()
        dhp = php2dhp(shp)
        preprocessor = self.create_preprocessor(dhp)
        estimator = self.create_estimator(dhp)
        pipeline = concat_pipeline(preprocessor, estimator)
        self.logger.debug(pipeline, pipeline[-1].hyperparams)
        return dhp, pipeline

    def hdl2phps(self, hdl: Dict):
        hdl2phps = HDL2SHPS()
        hdl2phps.set_task(self.ml_task)
        return hdl2phps(hdl)

    def parse_key(self, key: str):
        cnt = ""
        ix = 0
        for i, c in enumerate(key):
            if c.isdigit():
                cnt += c
            else:
                ix = i
                break
        cnt = int(cnt)
        key = key[ix:]
        pattern = re.compile(r"(\{.*\})")
        match = pattern.search(key)
        additional_info = {}
        if match:
            braces_content = match.group(1)
            _to = braces_content[1:-1]
            param_kvs = _to.split(",")
            for param_kv in param_kvs:
                k, v = param_kv.split("=")
                additional_info[k] = v
            key = pattern.sub("", key)
        if "->" in key:
            _from, _to = key.split("->")
            in_feature_groups = _from
            out_feature_groups = _to
        else:
            in_feature_groups, out_feature_groups = None, None
        if not in_feature_groups:
            in_feature_groups = None
        if not out_feature_groups:
            out_feature_groups = None
        return in_feature_groups, out_feature_groups, additional_info

    def create_preprocessor(self, dhp: Dict) -> Optional[GenericPipeline]:
        preprocessing_dict: dict = dhp["preprocessing"]
        pipeline_list = []
        for key, value in preprocessing_dict.items():
            name = key  # like: "cat->num"
            in_feature_groups, out_feature_groups, outsideEdge_info = self.parse_key(key)
            sub_dict = preprocessing_dict[name]
            if sub_dict is None:
                continue
            preprocessor = self.create_component(sub_dict, "preprocessing", name, in_feature_groups, out_feature_groups,
                                                 outsideEdge_info)
            pipeline_list.extend(preprocessor)
        if pipeline_list:
            return GenericPipeline(pipeline_list)
        else:
            return None

    def create_estimator(self, dhp: Dict) -> GenericPipeline:
        # 根据超参构造一个估计器
        return GenericPipeline(self.create_component(dhp["estimator"], "estimator", self.ml_task.role))

    def _create_component(self, key1, key2, params):
        cls = get_class_object_in_pipeline_components(key1, key2)
        component = cls()
        component.set_addition_info(self.addition_info)
        component.update_hyperparams(params)
        return component

    def create_component(self, sub_dhp: Dict, phase: str, step_name, in_feature_groups="all", out_feature_groups="all",
                         outsideEdge_info=None):
        pipeline_list = []
        assert phase in ("preprocessing", "estimator")
        packages = list(sub_dhp.keys())[0]
        params = sub_dhp[packages]
        packages = packages.split("|")
        grouped_params = group_dict_items_before_first_dot(params)
        if len(packages) == 1:
            if bool(grouped_params):
                grouped_params[packages[0]] = grouped_params.pop("single")
            else:
                grouped_params[packages[0]] = {}
        for package in packages[:-1]:
            preprocessor = self._create_component("preprocessing", package, grouped_params[package])
            preprocessor.in_feature_groups = in_feature_groups
            preprocessor.out_feature_groups = in_feature_groups
            pipeline_list.append([
                package,
                preprocessor
            ])
        key1 = "preprocessing" if phase == "preprocessing" else self.ml_task.mainTask
        component = self._create_component(key1, packages[-1], grouped_params[packages[-1]])
        component.in_feature_groups = in_feature_groups
        component.out_feature_groups = out_feature_groups
        if outsideEdge_info:
            component.update_hyperparams(outsideEdge_info)
        pipeline_list.append([
            step_name,
            component
        ])
        return pipeline_list

import inspect
import os
from typing import Dict, Optional, Callable

import numpy as np
from ConfigSpace import ConfigurationSpace
from frozendict import frozendict

from dsmac.facade.smac_hpo_facade import SMAC4HPO
from dsmac.scenario.scenario import Scenario
from hyperflow.constants import MLTask
from hyperflow.evaluation.ensemble_evaluator import EnsembleEvaluator
from hyperflow.evaluation.train_evaluator import TrainEvaluator
from hyperflow.hdl2shps.hdl2shps import HDL2SHPS
from hyperflow.manager.data_manager import DataManager
from hyperflow.manager.resource_manager import ResourceManager
from hyperflow.utils.concurrence import parse_n_jobs
from hyperflow.utils.config_space import get_random_initial_configs, get_grid_initial_configs
from hyperflow.utils.logging import get_logger


class Tuner():
    def __init__(
            self,
            evaluator: Callable = "TrainEvaluator",
            search_method: str = "smac",
            run_limit: int = 100,
            initial_runs: int = 20,
            search_method_params: dict = frozendict(),
            n_jobs: int = 1,
            exit_processes: Optional[int] = None
    ):
        self.logger = get_logger(__name__)
        search_method_params = dict(search_method_params)
        if isinstance(evaluator, str):
            if evaluator == "TrainEvaluator":
                evaluator = TrainEvaluator
            elif evaluator == "EnsembleEvaluator":
                evaluator = EnsembleEvaluator
            else:
                raise NotImplementedError
        assert callable(evaluator)
        self.evaluator_prototype = evaluator
        if inspect.isfunction(evaluator):
            self.evaluator = evaluator
        else:
            self.evaluator = evaluator()
        self.search_method_params = search_method_params
        assert search_method in ("smac", "grid", "random")
        if search_method in ("grid", "random"):
            initial_runs = 0
        self.initial_runs = initial_runs
        self.run_limit = run_limit
        self.evaluator = TrainEvaluator()  # todo 可选项
        self.search_method = search_method
        self.random_state = 0
        self.addition_info = {}
        self.resource_manager = None
        self.ml_task = None
        self.data_manager = None
        self.n_jobs = parse_n_jobs(n_jobs)
        if exit_processes is None:
            exit_processes = max(self.n_jobs // 3, 1)
        self.exit_processes = exit_processes

    def __str__(self):
        return (
            f"hyperflow.Tuner("
            f"evaluator={repr(self.evaluator_prototype.__name__)}, "
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

    # def set_addition_info(self, addition_info):
    #     self.addition_info = addition_info

    def hdl2shps(self, hdl: Dict):
        hdl2shps = HDL2SHPS()
        hdl2shps.set_task(self.ml_task)
        return hdl2shps(hdl)

    def set_hdl(self, hdl: Dict):
        self.hdl = hdl
        # todo: 泛化ML管线后，可能存在多个preprocessing
        self.shps: ConfigurationSpace = self.hdl2shps(hdl)
        self.shps.seed(self.random_state)

    def set_resource_manager(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.evaluator.set_resource_manager(resource_manager)

    def set_task(self, ml_task: MLTask):
        self.ml_task = ml_task

    def set_data_manager(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.ml_task = data_manager.ml_task

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
            initial_configs,
            evaluator_params=frozendict(),
            instance_id="",
            rh_db_type="sqlite",
            rh_db_params=frozendict(),
            rh_db_table_name="runhistory"
    ):
        # time.sleep(random.random())
        if not initial_configs:
            self.logger.warning("Haven't initial_configs. Return.")
            return

        self.evaluator.init_data(**evaluator_params)

        self.scenario = Scenario(
            {
                "run_obj": "quality",
                "runcount-limit": 1000,
                "cs": self.shps,  # configuration space
                "deterministic": "true",
                "instances": [[instance_id]]
                # todo : 如果是local，存在experiment，如果是其他文件系统，不输出smac
                # "output_dir": self.resource_manager.smac_output_dir,
            },
            initial_runs=0,
            db_type=rh_db_type,
            db_params=rh_db_params,
            db_table_name=rh_db_table_name,
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

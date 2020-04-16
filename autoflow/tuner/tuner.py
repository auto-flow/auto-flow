import inspect
import os
from typing import Dict, Optional, Callable, Union

import numpy as np
from ConfigSpace import ConfigurationSpace
from frozendict import frozendict

from dsmac.facade.smac_hpo_facade import SMAC4HPO
from dsmac.scenario.scenario import Scenario
from autoflow.evaluation.ensemble_evaluator import EnsembleEvaluator
from autoflow.evaluation.train_evaluator import TrainEvaluator
from autoflow.hdl2shps.hdl2shps import HDL2SHPS
from autoflow.manager.data_manager import DataManager
from autoflow.manager.resource_manager import ResourceManager
from autoflow.utils.concurrence import parse_n_jobs
from autoflow.utils.config_space import get_random_initial_configs, get_grid_initial_configs
from autoflow.utils.klass import StrSignatureMixin
from autoflow.utils.logging import get_logger
from autoflow.utils.ml_task import MLTask

class Tuner(StrSignatureMixin):
    '''
    ``Tuner`` if class who agent an abstract search process.
    '''
    def __init__(
            self,
            evaluator: Union[Callable, str] = "TrainEvaluator",
            search_method: str = "smac",
            run_limit: int = 100,
            initial_runs: int = 20,
            search_method_params: dict = frozendict(),
            n_jobs: int = 1,
            exit_processes: Optional[int] = None,
            limit_resource: bool = True,
            per_run_time_limit: float = 60,
            per_run_memory_limit: float = 3072,
            time_left_for_this_task: float = None,
            debug=False
    ):
        '''

        Parameters
        ----------
        evaluator: callable, str
            ``evaluator`` is a function or callable class (implement magic method ``__call__``) or string-indicator.

            ``evaluator`` can receive a shp(SMAC Hyper Param, :class:`ConfigSpace.ConfigurationSpace`),

            and return a dict ,which contains such keys:

                * ``loss``, you can think of it as negative reward.
                * ``status``, a string , ``SUCCESS`` means fine, ``FAILED`` means crashed.

            As default,  "TrainEvaluator" is the string-indicator of :class:`autoflow.evaluation.train_evaluator.TrainEvaluator` .

        search_method: str
            Specific searching method, ``random``, ``smac``, ``grid`` are available.

                * ``random`` Random Search Algorithm,
                * ``grid``   Grid   Search Algorithm,
                * ``smac``   Bayes Search by SMAC Algorithm.

        run_limit: int
            Limitation of running step.

        initial_runs: int
            If you choose ``smac`` algorithm,

            you should realize the SMAC algorithm has a initialize procedure,

            The algorithm needs enough initial runs to get enough experience.

            This param will be omitted if ``random`` or ``grid`` is selected.

        search_method_params: dict
            Configuration for specific search method.

        n_jobs: int
            ``n_jobs`` searching process will start.

        exit_processes: int
        limit_resource: bool
            If ``limit_resource = True``, a searching trial will be killed if it use more CPU times or memory.

        per_run_time_limit: float
            will active if ``limit_resource = True``.

            a searching trial will be killed if it use CPU times more than ``per_run_time_limit``.

        per_run_memory_limit: float
            will active if ``limit_resource = True``.

            a searching trial will be killed if it use memory more than ``per_run_memory_limit``.

        time_left_for_this_task: float
            will active if ``limit_resource = True``.

            a searching task will be killed if it's totally run time more than ``time_left_for_this_task``.

        debug: bool
            For debug mode.

            Exception will be re-raised if ``debug = True``
        '''
        self.debug = debug
        self.per_run_memory_limit = per_run_memory_limit
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.limit_resource = limit_resource
        self.logger = get_logger(self)
        if self.debug and self.limit_resource:
            self.logger.warning(
                "Tuner.debug and Tuner.limit_resource cannot be both True. set Tuner.limit_resource to False.")
            self.limit_resource = False
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
        self.evaluator.debug = self.debug
        self.search_method_params = search_method_params
        assert search_method in ("smac", "grid", "random")
        if search_method in ("grid", "random"):
            initial_runs = 0
        self.initial_runs = initial_runs
        self.run_limit = run_limit
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
        # self.evaluator.set_resource_manager(resource_manager)

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
        senario_dict = {
            "run_obj": "quality",
            "runcount-limit": 1000,
            "cs": self.shps,  # configuration space
            "deterministic": "true",
            "instances": [[instance_id]],
            "cutoff_time": self.per_run_time_limit,
            "memory_limit": self.per_run_memory_limit
            # todo : 如果是local，存在experiment，如果是其他文件系统，不输出smac
            # "output_dir": self.resource_manager.smac_output_dir,
        }
        self.scenario = Scenario(
            senario_dict,
            initial_runs=0,
            db_type=rh_db_type,
            db_params=rh_db_params,
            db_table_name=rh_db_table_name,
            anneal_func=self.search_method_params.get("anneal_func"),
            use_pynisher=self.limit_resource
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

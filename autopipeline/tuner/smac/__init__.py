from typing import Dict

import numpy as np

from autopipeline.data.xy_data_manager import XYDataManager
from autopipeline.evaluation.train_evaluator import TrainEvaluator
from autopipeline.hdl2phps.smac_hdl2phps import SmacHDL2PHPS
from autopipeline.metrics import Scorer
from autopipeline.php2dhp.smac_php2dhp import SmacPHP2DHP
from autopipeline.tuner.base import PipelineTuner
from autopipeline.utils.pipeline import concat_pipeline
from dsmac.distributer import SingleDistributer
from dsmac.facade.smac_hpo_facade import SMAC4HPO
from dsmac.scenario.scenario import Scenario


class SmacPipelineTuner(PipelineTuner):
    def __init__(
            self,
            runcount_limit: int = 50,
            initial_runs: int = 20,
            random_state: int = 42,
            evaluator: TrainEvaluator = None,
            distributer=SingleDistributer(n_jobs=1),
    ):
        super(SmacPipelineTuner, self).__init__(
            runcount_limit,
            initial_runs,
            random_state,
            evaluator,
        )

        self.distributer = distributer

        self.evaluator.set_php2model(self.php2model)

        # 将smac中的参数迁移过来
        # 与训练任务有关的一些参数（如分类回归任务，数据集（用于初始化所有算法模型的默认超参））
        # 思考不同的优化策略？

    def run(
            self,
            datamanager: XYDataManager,
            metric: Scorer,
            all_scoring_functions: bool,
            splitter,
            smac_output_dir
    ):
        if hasattr(splitter, "random_state"):
            setattr(splitter, "random_state", self.random_state)
        self.set_task(datamanager.task)
        self.evaluator.init_data(
            datamanager,
            metric,
            all_scoring_functions,
            splitter,
        )
        # todo: metalearn

        self.scenario = Scenario(
            {
                "run_obj": "quality",
                "runcount-limit": self.runcount_limit,
                "cs": self.phps,  # configuration space
                "deterministic": "true",
                "output_dir": smac_output_dir
            },
            distributer=self.distributer,
            initial_runs=self.initial_runs,
            after_run_callback=self.evaluator.resource_manager.delete_models
        )
        # todo 将 file_system 传入，或者给file_system添加 runtime 参数
        smac = SMAC4HPO(
            scenario=self.scenario,
            rng=np.random.RandomState(self.random_state),
            tae_runner=self.evaluator,
            initial_configurations=self.phps.get_default_configuration()
        )
        self.incumbent = smac.optimize()
        # todo: ensemble

    def php2model(self, php):
        php2dhp = SmacPHP2DHP()
        dhp = php2dhp(php)
        preprocessor = self.create_preprocessor(dhp)
        estimator = self.create_estimator(dhp)
        pipeline = concat_pipeline(preprocessor, estimator)
        print(pipeline, pipeline[-1].hyperparams)
        return dhp,pipeline

    def hdl2phps(self, hdl: Dict):
        hdl2phps = SmacHDL2PHPS()
        return hdl2phps(hdl)

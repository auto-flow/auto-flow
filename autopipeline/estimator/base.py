from typing import Union, List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from autopipeline.data.xy_data_manager import XYDataManager
from autopipeline.hdl.default_hp import add_public_info_to_default_hp
from autopipeline.hdl.hdl_constructor import HDL_Constructor
from autopipeline.metrics import r2, accuracy
from autopipeline.tuner.base import PipelineTuner
from autopipeline.tuner.smac import SmacPipelineTuner
from autopipeline.utils.resource_manager import ResourceManager


class AutoPipelineEstimator(BaseEstimator):

    def __init__(
            self,
            tuner: PipelineTuner = None,  # 抽象化的优化的全过程
            hdl_constructor: HDL_Constructor = None,  # 用户自定义初始超参
            resource_manager: ResourceManager = None
    ):
        if not tuner:
            tuner = SmacPipelineTuner()
        self.tuner: SmacPipelineTuner = tuner
        if not hdl_constructor:
            hdl_constructor = HDL_Constructor()
        if isinstance(hdl_constructor, dict):
            print("使用用户自定义超参描述语言")
        self.hdl_constructor = hdl_constructor
        self.random_state = tuner.random_state
        if not resource_manager:
            self.resource_manager = ResourceManager()

    def fit(
            self, X: np.ndarray, y,
            metric=None,
            X_test=None, y_test=None,
            dataset_name="default_dataset_name",
            all_scoring_functions=False,
            splitter=KFold(5, True, 42),
            feature_groups: Union[None, str, List] = None
    ):
        self.data_manager = XYDataManager(  # todo: 将feature groups 纳入 data_manager 管理
            X, y, X_test, y_test, dataset_name, feature_groups
        )
        self.hdl_constructor.set_data_manager(self.data_manager)
        self.hdl_constructor.run()
        self.hdl = self.hdl_constructor.get_hdl()
        self.default_hp = self.hdl_constructor.get_default_hp()
        self.tuner.set_data_manager(self.data_manager)
        self.tuner.set_hdl(self.hdl)
        self.tuner.set_addition_info({})  # {"shape": X.shape}
        self.tuner.evaluator.set_resource_manager(self.resource_manager)
        # todo : 增加 n_jobs ? 调研默认值
        add_public_info_to_default_hp(
            self.default_hp, {"random_state": self.random_state}
        )
        self.tuner.set_default_hp(self.default_hp)
        self.task = self.data_manager.task
        if metric is None:
            if self.task.mainTask == "regression":
                metric = r2
            elif self.task.mainTask == "classification":
                metric = accuracy
            else:
                raise NotImplementedError()
        # todo: 根据datamanager获取的task信息构造HDL
        self.tuner.run(
            self.data_manager,
            metric,
            all_scoring_functions,
            splitter,
            self.resource_manager.smac_output_dir
        )
        return self

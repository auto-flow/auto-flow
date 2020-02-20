from typing import Dict
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from autopipeline.data.xy_data_manager import XYDataManager
from autopipeline.init_data import init_all
from autopipeline.tuner.base import PipelineTuner
from autopipeline.metrics import accuracy

class AutoPipelineEstimator(BaseEstimator):

    def __init__(
            self,
            tuner:PipelineTuner, # 抽象化的优化的全过程
            custom_init_param=None,   # 用户自定义初始超参
            custom_hyper_param=None,  # 用户自定义超参
    ):
        self.custom_hyper_param = custom_hyper_param
        self.custom_init_param = custom_init_param
        self.tuner = tuner
        if self.custom_hyper_param:
            self.tuner.set_phps(self.custom_hyper_param)
        else:
            pass
            # todo: 根据具体的任务装配一个默认的管道


    def fit(
            self,X:np.ndarray,y,
            metric=accuracy,
            X_test=None,y_test=None,
            dataset_name="default_dataset_name",
            all_scoring_functions=False,
            spliter=KFold(5,True,42)
    ):
        init_all(
            self.custom_init_param,
            {"shape": X.shape},
            {'random_state':self.tuner.random_state}
        )
        self.datamanager=XYDataManager(
            X,y,X_test,y_test,None,dataset_name
        )
        self.tuner.run(
            self.datamanager,
            metric,
            all_scoring_functions,
            spliter
        )
        return self


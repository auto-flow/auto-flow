import os
import time
import warnings
from typing import Dict, Callable

import numpy as np

from autopipeline.constants import Task
from autopipeline.data.xy_data_manager import XYDataManager
from autopipeline.metrics import calculate_score, CLASSIFICATION_METRICS, Scorer
from autopipeline.utils.logging_ import get_logger

__all__ = [
    'AbstractEvaluator'
]



class AbstractEvaluator(object):


    def init_data(
            self,
            datamanager:XYDataManager,
            metric:Scorer,
            all_scoring_functions:bool,
            spliter=None,
    ):
        self.spliter=spliter
        # self.datamanager = self.backend.load_datamanager()
        self.datamanager=datamanager
        self.X_train = self.datamanager.data['X_train']
        self.y_train = self.datamanager.data['Y_train']
        self.X_valid = self.datamanager.data.get('X_valid')
        self.y_valid = self.datamanager.data.get('Y_valid')
        self.X_test = self.datamanager.data.get('X_test')
        self.y_test = self.datamanager.data.get('Y_test')

        self.metric = metric
        self.task_type:Task = self.datamanager.task
        # self.seed = seed

        # self.output_y_hat_optimization = output_y_hat_optimization
        self.all_scoring_functions = all_scoring_functions
        # self.disable_file_output = disable_file_output

        if self.task_type.mainTask=="regression":
            self.predict_function = self._predict_regression
        else:
            self.predict_function = self._predict_proba

        categorical_mask = []
        for feat in self.datamanager.feat_type:
            if feat.lower() == 'numerical':
                categorical_mask.append(False)
            elif feat.lower() == 'categorical':
                categorical_mask.append(True)
            else:
                raise ValueError(feat)

        # self.subsample = subsample

        logger_name = self.__class__.__name__
        self.logger = get_logger(logger_name)

        self.Y_optimization = None
        self.Y_actual_train = None



    def _loss(self, y_true, y_hat):
        all_scoring_functions = (
            self.all_scoring_functions
            if self.all_scoring_functions is None
            else self.all_scoring_functions
        )

        score = calculate_score(
            y_true, y_hat, self.task_type, self.metric,
            all_scoring_functions=all_scoring_functions)

        if hasattr(score, '__len__'):
            # TODO: instead of using self.metric, it should use all metrics given by key.
            # But now this throws error...
            # FIXME： Regression  ?
            err = {key: metric._optimum - score[key] for key, metric in
                   CLASSIFICATION_METRICS.items() if key in score}
        else:
            err = self.metric._optimum - score

        return err

    def loss(self,y_true, y_hat):
        err=self._loss(y_true, y_hat)
        if isinstance(err,dict):
            # todo: 做记录
            return err[self.metric.name]
        return err


    def _predict_proba(self, X, model):
        Y_pred = model.predict_proba(X)
        return Y_pred

    def _predict_regression(self, X, model):
        Y_pred = model.predict(X)
        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape((-1, 1))
        return Y_pred

    def get_Xy(self):
        raise NotImplementedError()

    def evaluate(self,model,X,y):
        raise NotImplementedError()

    def __call__(self, php:Dict):
        # 1. 将php变成model
        # model=self.php2model(php)
        # 2. 获取数据
        X,y= self.get_Xy()
        # 3. 进行评价
        return self.evaluate(php,X,y)





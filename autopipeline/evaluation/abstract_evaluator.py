from time import time
from typing import Dict

from autopipeline.constants import Task
from autopipeline.data.xy_data_manager import XYDataManager
from autopipeline.metrics import calculate_score, CLASSIFICATION_METRICS, Scorer
from autopipeline.utils.logging_ import get_logger
from autopipeline.utils.resource_manager import ResourceManager

__all__ = [
    'AbstractEvaluator'
]


class AbstractEvaluator(object):

    def init_data(
            self,
            data_manager: XYDataManager,
            metric: Scorer,
            all_scoring_functions: bool,
            splitter=None,
    ):
        self.splitter = splitter
        self.data_manager = data_manager
        self.X_train = self.data_manager.data['X_train']
        self.y_train = self.data_manager.data['y_train']
        self.X_test = self.data_manager.data.get('X_test')
        self.y_test = self.data_manager.data.get('y_test')

        self.metric = metric
        self.task: Task = self.data_manager.task
        # self.seed = seed

        # self.output_y_hat_optimization = output_y_hat_optimization
        self.all_scoring_functions = all_scoring_functions
        # self.disable_file_output = disable_file_output

        if self.task.mainTask == "regression":
            self.predict_function = self._predict_regression
        else:
            self.predict_function = self._predict_proba

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
            y_true, y_hat, self.task, self.metric,
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

    def set_resource_manager(self, resource_manager: ResourceManager):
        self._resouce_manager = resource_manager

    @property
    def resource_manager(self):
        return self._resouce_manager

    def loss(self, y_true, y_hat):
        err = self._loss(y_true, y_hat)
        if isinstance(err, dict):
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

    def evaluate(self, model, X, y):
        raise NotImplementedError()

    def set_php2model(self, php2model):
        self.php2model = php2model

    def __call__(self, php: Dict):
        # 1. 将php变成model
        trial_id = getattr(php, "trial_id")
        start=time()
        dhp, model = self.php2model(php)
        # 2. 获取数据
        X, y = self.get_Xy()
        # 3. 进行评价
        loss, info = self.evaluate(model, X, y)  # todo : 考虑失败的情况
        # 4. 持久化
        cost_time=time()-start
        info["php"] = php
        info["dhp"] = dhp
        estimator = list(dhp.get("MHP", {"unk": ""}).keys())[0]
        info["estimator"] = estimator
        info["trial_id"] = trial_id
        info["cost_time"] = cost_time
        self.resource_manager.persistent_evaluated_model(info)
        # 记录必要的信息，用于后续删除表现差的模型
        self.resource_manager.insert_to_db(trial_id, estimator, loss,cost_time)
        # 向数据库insert一条记录，避免互斥写文件?
        return loss

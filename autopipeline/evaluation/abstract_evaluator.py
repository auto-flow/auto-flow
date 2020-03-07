from time import time

from ConfigSpace.configuration_space import Configuration

from autopipeline.constants import Task
from autopipeline.data.xy_data_manager import XYDataManager
from autopipeline.metrics import calculate_score, CLASSIFICATION_METRICS, Scorer, REGRESSION_METRICS
from autopipeline.utils.logging_ import get_logger
from autopipeline.utils.resource_manager import ResourceManager
from dsmac.runhistory.utils import get_id_of_config

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

    def loss(self, y_true, y_hat):
        all_scoring_functions = (
            self.all_scoring_functions
            if self.all_scoring_functions is None
            else self.all_scoring_functions
        )

        score = calculate_score(
            y_true, y_hat, self.task, self.metric,
            all_scoring_functions=all_scoring_functions)

        if isinstance(score, dict):
            err = self.metric._optimum - score[self.metric.name]
            all_score = score
        elif isinstance(score, (int, float)):
            err = self.metric._optimum - score
            all_score = None
        else:
            raise TypeError

        return err, all_score

    def set_resource_manager(self, resource_manager: ResourceManager):
        self._resouce_manager = resource_manager

    @property
    def resource_manager(self):
        return self._resouce_manager

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

    def evaluate(self, model, X_train, y_train,X_test,y_test):
        raise NotImplementedError()

    def set_php2model(self, php2model):
        self.php2model = php2model

    def __call__(self, php: Configuration):
        # 1. 将php变成model
        trial_id = get_id_of_config(php)
        start = time()
        dhp, model = self.php2model(php)
        # 2. 获取数据
        X_train, y_train,X_test,y_test = self.get_Xy()
        # 3. 进行评价
        loss, info = self.evaluate(model, X_train, y_train,X_test,y_test)  # todo : 考虑失败的情况
        # 4. 持久化
        cost_time = time() - start
        info["trial_id"] = trial_id
        info["status"] = "success"
        info["program_hyper_param"] = php
        info["dict_hyper_param"] = dhp
        estimator = list(dhp.get("MHP", {"unk": ""}).keys())[0]
        info["estimator"] = estimator
        info["trial_id"] = trial_id
        info["cost_time"] = cost_time
        self.resource_manager.insert_to_db(info)
        return loss

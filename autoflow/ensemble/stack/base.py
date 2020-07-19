from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression, ElasticNet

from autoflow.ensemble.base import EnsembleEstimator
from autoflow.utils import typing_
from autoflow.utils.logging_ import get_logger


class StackEstimator(EnsembleEstimator):

    def __init__(
            self,
            meta_learner=None,
            use_features_in_secondary=False,
    ):
        self.use_features_in_secondary = use_features_in_secondary
        assert self.mainTask in ("classification", "regression")
        if not meta_learner:
            if self.mainTask == "classification":
                meta_learner = LogisticRegression(
                    penalty='elasticnet',
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    fit_intercept=False
                )
            elif self.mainTask == "regression":
                meta_learner = ElasticNet(fit_intercept=False,random_state=10)
        self.meta_learner = meta_learner
        self.logger=get_logger(self)

    def fit(self, X, y):
        # fixme: 2020-4-9 更新后， 此方法弃用
        # todo ： 验证所有的 y_true_indexes 合法
        # todo : 做完stack之后在验证集上的表现
        meta_features = self.predict_meta_features(X, True)
        self.meta_learner.fit(meta_features, y)

    def fit_trained_data(
            self,
            estimators_list: List[List[typing_.GenericEstimator]],
            y_preds_list: List[List[np.ndarray]],
            y_true_indexes_list: List[List[np.ndarray]],
            y_true: np.ndarray
    ):
        super(StackEstimator, self).fit_trained_data(estimators_list, y_preds_list, y_true_indexes_list, y_true)
        meta_features = self.predict_meta_features(None, True)
        # todo: 对元学习器做 automl
        self.meta_learner.fit(meta_features, self.stacked_y_true)
        score=self.meta_learner.score(meta_features, self.stacked_y_true)
        self.logger.info(f"meta_learner's performance: {score}")
        self.logger.info(f"meta_learner's coefficient: {self.meta_learner.coef_}")



    def predict_meta_features(self, X, is_train):
        raise NotImplementedError

    def _do_predict(self, X, predict_fn):
        meta_features = self.predict_meta_features(X, False)
        return predict_fn(meta_features)

    def predict(self, X):
        return self._do_predict(X, self.meta_learner.predict)

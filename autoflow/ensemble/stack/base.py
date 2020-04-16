from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression, ElasticNet

from autoflow.ensemble.base import EnsembleEstimator
from autoflow.utils import typing


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
                meta_learner = LogisticRegression(penalty='l2', solver="lbfgs", multi_class="auto",
                                                  random_state=10)
            elif self.mainTask == "regression":
                meta_learner = ElasticNet()
        self.meta_learner = meta_learner

    def fit(self, X, y):
        # fixme: 2020-4-9 更新后， 此方法弃用
        # todo ： 验证所有的 y_true_indexes 合法
        # todo : 做完stack之后在验证集上的表现
        meta_features = self.predict_meta_features(X, True)
        self.meta_learner.fit(meta_features, y)

    def fit_trained_data(
            self,
            estimators_list: List[List[typing.GenericEstimator]],  # fixme : typing是否存在局限性？
            y_preds_list: List[List[np.ndarray]],
            y_true_indexes_list: List[List[np.ndarray]],
            y_true: np.ndarray
    ):
        super(StackEstimator, self).fit_trained_data(estimators_list, y_preds_list, y_true_indexes_list, y_true)
        meta_features = self.predict_meta_features(None, True)
        self.meta_learner.fit(meta_features, y_true)

    def predict_meta_features(self, X, is_train):
        raise NotImplementedError

    def _do_predict(self, X, predict_fn):
        meta_features = self.predict_meta_features(X, False)
        return predict_fn(meta_features)

    def predict(self, X):
        return self._do_predict(X, self.meta_learner.predict)

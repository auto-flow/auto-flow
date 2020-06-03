from typing import List

import numpy as np
from sklearn.base import BaseEstimator

from autoflow.data_container.base import get_container_data
from autoflow.utils import typing_


class EnsembleEstimator(BaseEstimator):
    mainTask = None

    def build_prediction_list(self):
        prediction_list = []
        assert len(self.y_true_indexes_list) > 1
        # splitter 的 random_state都是相同的， 所以认为  y_true_indexes_list 的每个 y_true_indexes 都相同
        assert not np.any(np.array([np.hstack(y_true_indexes) for y_true_indexes in  self.y_true_indexes_list]).var(axis=0))
        for y_preds in self.y_preds_list:
            prediction_list.append(np.concatenate(y_preds))  # concat in axis 0
        self.prediction_list = prediction_list
        y_true_indexes = self.y_true_indexes_list[0]
        self.stacked_y_true = self.y_true[np.hstack(y_true_indexes)]
        assert self.prediction_list[0].shape[0] == self.stacked_y_true.shape[0]

    def fit_trained_data(
            self,
            estimators_list: List[List[typing_.GenericEstimator]],
            y_true_indexes_list: List[List[np.ndarray]],
            y_preds_list: List[List[np.ndarray]],
            y_true: np.ndarray
    ):
        self.y_preds_list = y_preds_list
        self.y_true_indexes_list = y_true_indexes_list
        self.estimators_list = estimators_list
        self.y_true = get_container_data(y_true)
        self.build_prediction_list()

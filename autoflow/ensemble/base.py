from collections import Counter
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
        self.y_true_index_list = [np.hstack(y_true_indexes) for y_true_indexes in self.y_true_indexes_list]
        size_list = [y.shape[0] for y in self.y_true_index_list]
        size_counter = Counter(size_list)
        if len(size_counter) > 1:
            valid_size = size_counter.most_common(1)[0][0]
            y_true_index_list_ = []
            y_preds_list_ = []
            for size, y_true_index, y_preds in zip(size_list, self.y_true_index_list, self.y_preds_list):
                if size == valid_size:
                    y_true_index_list_.append(y_true_index)
                    y_preds_list_.append(y_preds)
            self.y_true_index_list = y_true_index_list_
            self.y_preds_list = y_preds_list_
        elif len(size_counter) == 0:
            raise ValueError
        assert not np.any(np.array(self.y_true_index_list).var(axis=0))
        for y_preds in self.y_preds_list:
            prediction_list.append(np.concatenate(y_preds))  # concat in axis 0
        self.prediction_list = prediction_list
        self.stacked_y_true = self.y_true[self.y_true_index_list[0]]
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

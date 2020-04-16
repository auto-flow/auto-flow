from typing import List

import numpy as np
from sklearn.base import BaseEstimator

from autoflow.utils import typing


class EnsembleEstimator(BaseEstimator):
    mainTask = None


    def build_prediction_list(self):
        prediction_list = []
        for y_true_indexes, y_preds in zip(self.y_true_indexes_list, self.y_preds_list):
            prediction = np.zeros_like(np.vstack(y_preds))
            for y_index, y_pred in zip(y_true_indexes, y_preds):
                prediction[y_index] = y_pred
            prediction_list.append(prediction)
        self.prediction_list = prediction_list

    def fit_trained_data(
            self,
            estimators_list: List[List[typing.GenericEstimator]],
            y_true_indexes_list: List[List[np.ndarray]],
            y_preds_list: List[List[np.ndarray]],
            y_true: np.ndarray
    ):
        self.y_preds_list = y_preds_list
        self.y_true_indexes_list = y_true_indexes_list
        self.estimators_list = estimators_list
        self.build_prediction_list()
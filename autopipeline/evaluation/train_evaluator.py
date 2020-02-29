from typing import Callable

import numpy as np

from autopipeline.data.xy_data_manager import XYDataManager
from autopipeline.evaluation.abstract_evaluator import AbstractEvaluator
from autopipeline.metrics import Scorer


class TrainEvaluator(AbstractEvaluator):




    def get_Xy(self):
        return self.X_train, self.y_train

    def evaluate(self, model, X, y):

        # splitter 必须存在
        losses = []
        models=[]
        indices=[]
        for train_index, test_index in self.splitter.split(X, y):
            X_train, X_test = self.X_train[train_index], self.X_train[test_index]
            y_train, y_test = self.y_train[train_index], self.y_train[test_index]
            fitted_model=model.fit(X_train, y_train)
            models.append(fitted_model)
            indices.append([train_index,test_index])
            y_pred = self.predict_function(X_test, model)
            loss = self.loss(y_test, y_pred)
            losses.append(loss)
        final_loss= np.array(losses).mean()
        return final_loss

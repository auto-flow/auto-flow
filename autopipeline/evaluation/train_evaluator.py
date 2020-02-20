from typing import Callable

import numpy as np

from autopipeline.data.xy_data_manager import XYDataManager
from autopipeline.evaluation.abstract_evaluator import AbstractEvaluator
from autopipeline.metrics import Scorer


class TrainEvaluator(AbstractEvaluator):


    def init_php2model(self,php2model):
        self.php2model=php2model

    def get_Xy(self):
        return self.X_train, self.y_train

    def evaluate(self, php, X, y):
        if self.spliter:
            losses = []
            models=[]
            indices=[]
            for train_index, test_index in self.spliter.split(X, y):
                X_train, X_test = self.X_train[train_index], self.X_train[test_index]
                y_train, y_test = self.y_train[train_index], self.y_train[test_index]
                model = self.php2model(php)
                model.fit(X_train, y_train)
                models.append(model)
                indices.append([train_index,test_index])
                y_pred = self.predict_function(X_test, model)
                loss = self.loss(y_test, y_pred)
                losses.append(loss)
            final_loss= np.array(losses).mean()
        elif self.X_valid:
            model = self.php2model(php)
            model.fit(self.X_train, self.y_train)
            y_pred = self.predict_function(self.X_valid, model)
            final_loss = self.loss(self.y_valid, y_pred)
            losses=[final_loss]
            models=[model]
            indices=[None]
            # todo 把这些数据存下来
        else:
            raise NotImplementedError()
        print(final_loss)
        return final_loss

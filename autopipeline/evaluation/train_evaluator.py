import numpy as np

from autopipeline.evaluation.abstract_evaluator import AbstractEvaluator


class TrainEvaluator(AbstractEvaluator):

    def get_Xy(self):
        return self.X_train, self.y_train

    def evaluate(self, model, X, y):
        # splitter 必须存在
        losses = []
        models = []
        y_test_indices = []
        y_preds = []
        # todo: indices是否有存在的必要，是否冗余了？
        for train_index, test_index in self.splitter.split(X, y):
            X_train, X_test = self.X_train[train_index], self.X_train[test_index]
            y_train, y_test = self.y_train[train_index], self.y_train[test_index]
            fitted_model = model.fit(X_train, y_train)
            models.append(fitted_model)
            y_test_indices.append(test_index)
            y_pred = self.predict_function(X_test, model)
            y_preds.append(y_pred)
            loss = self.loss(y_test, y_pred)  # todo: 获取多组metrics的情况
            losses.append(loss)
        final_loss = np.array(losses).mean()
        info = {
            "loss":final_loss,
            "losses": losses,
            "models": models,
            "y_test_indices": y_test_indices,
            "y_preds": y_preds

        }

        return final_loss, info

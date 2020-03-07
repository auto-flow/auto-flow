from collections import defaultdict
from contextlib import redirect_stderr
from io import StringIO

import numpy as np

from autopipeline.ensemble.mean.regressor import MeanRegressor
from autopipeline.ensemble.vote.classifier import VoteClassifier
from autopipeline.evaluation.abstract_evaluator import AbstractEvaluator


class TrainEvaluator(AbstractEvaluator):

    def get_Xy(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

    def evaluate(self, model, X, y, X_test, y_test):
        warning_info = StringIO()
        with redirect_stderr(warning_info):
            # splitter 必须存在
            losses = []
            models = []
            y_trues = []
            y_preds = []
            all_scores = []
            for train_index, valid_index in self.splitter.split(X, y):
                X_train, X_valid = X[train_index], X[valid_index]
                y_train, y_valid = y[train_index], y[valid_index]
                fitted_model = model.fit(X_train, y_train)
                models.append(fitted_model)
                y_trues.append(y[valid_index])
                y_pred = self.predict_function(X_valid, model)
                y_preds.append(y_pred)
                loss, all_score = self.loss(y_valid, y_pred)
                losses.append(float(loss))
                all_scores.append(all_score)

            final_loss = float(np.array(losses).mean())
            if len(all_scores) > 0 and all_scores[0]:
                all_score = defaultdict(list)
                for cur_all_score in all_scores:
                    assert isinstance(cur_all_score, dict)
                    for key, value in cur_all_score.items():
                        all_score[key].append(value)
                for key in all_score.keys():
                    all_score[key] = float(np.mean(all_score[key]))
            else:
                all_score = None
                all_scores = None
            info = {
                "loss": final_loss,
                "losses": losses,
                "all_score": all_score,
                "all_scores": all_scores,
                "models": models,
                "y_trues": y_trues,
                "y_preds": y_preds,
            }
            if y_test is not None:
                # 验证集训练模型的组合去预测测试集的数据
                if self.task.mainTask == "classification":
                    trainset_estimator = VoteClassifier(models)
                    y_test_pred = trainset_estimator.predict_proba(X_test)
                else:
                    trainset_estimator = MeanRegressor(models)
                    y_test_pred = trainset_estimator.predict(X_test)
                test_loss, test_all_score = self.loss(y_test, y_test_pred)
                info.update({
                    "test_loss": test_loss,
                    "test_all_score": test_all_score,
                    "y_test_true": y_test,
                    "y_test_pred": y_test_pred
                })
        info["warning_info"] = warning_info.getvalue()
        return final_loss, info

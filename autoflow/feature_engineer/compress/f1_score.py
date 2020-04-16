import numpy as np

from autoflow.feature_engineer.compress.similarity_base import SimilarityBase


def valid(arr):
    unique = np.unique(arr)
    if unique.size > 2:
        return False
    elif unique.size == 2 and np.all(unique == np.array([0, 1])):
        return True
    elif unique.size == 1 and unique[0] in [0, 1]:
        return True
    else:
        return False


def f1_score(y_true, y_pred):
    y_pred_sum = np.sum(y_pred)
    y_true_sum = np.sum(y_true)
    if y_pred_sum == 0 or y_true_sum == 0:
        return 0
    P = np.sum(y_pred[y_pred == y_true]) / y_pred_sum
    R = np.sum(y_true[y_pred == y_true]) / y_true_sum
    if P == 0 or R == 0:
        return 0
    return 2 / (1 / P + 1 / R)

class F1Score(SimilarityBase):
    name = "f1_score"

    def core_func(self, s, e, L):
        # X_是全局变量
        to_del = []
        for i in range(s, e):
            for j in range(i + 1, L):
                if not (valid(self.X_[:, i]) and valid(self.X_[:, j])):
                    continue
                r = f1_score(self.X_[:, i], self.X_[:, j])
                if r > self.threshold:
                    to_del.append([r, i])
                    break
        return to_del
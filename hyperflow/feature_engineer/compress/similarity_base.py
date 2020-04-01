from time import time
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def get_start_end_tuples(threads, L):
    N = L * (L - 1) // 2
    chunks = N // threads
    start = 0
    span_list = []
    for i in range(threads):
        ans = ((4 * (start ** 2) - 4 * start + 8 * chunks + 1) ** 0.5 - 2 * start + 1) / 2
        n_ = round(ans)
        end = min(L, start + n_)
        span_list.append(end - start)
        start = end
    start = end = 0
    start_end_tuples = []
    for span in reversed(span_list):
        end = start + span
        start_end_tuples.append([start, end])
        start = end
    print(start_end_tuples)
    return start_end_tuples


class SimilarityBase(TransformerMixin, BaseEstimator):
    def __init__(self, threshold, n_jobs=1, max_delete=1):
        self.max_delete = max_delete
        self.to_delete = []
        self.threshold = threshold
        self.n_jobs = n_jobs
        self._type = "DataFrame"

    name = None

    def core_func(self, s, e, L):
        raise NotImplementedError()

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            self._type = "ndarray"
        start = time()
        self.X_ = X.values  # X is DataFrame
        L = self.X_.shape[1]
        split_points = get_start_end_tuples(self.n_jobs, L)
        with joblib.parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            ans = joblib.Parallel()(
                joblib.delayed(self.core_func)(s, e, L)
                for s, e in split_points
            )
        to_del = ans[0]
        for other in ans[1:]:
            to_del.extend(other)
        self.to_delete = []
        to_del.sort(key=lambda x: x[0], reverse=True)
        for p, ix in to_del:
            self.to_delete.append(X.columns[ix])
        self.to_delete = self.to_delete[:int(X.shape[1] * self.max_delete)]
        end = time()
        print("use time:", end - start)
        return self

    def transform(self, X, y=None):
        _type = "DataFrame"
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            _type = "ndarray"
        assert self._type == _type
        col_before = X.shape[1]
        X = X.drop(self.to_delete, axis=1)
        col_after = X.shape[1]
        print(f"features were deleted by {self.name}")
        print(f"features before {col_before} , after {col_after}, {col_before - col_after} were deleted")
        return X

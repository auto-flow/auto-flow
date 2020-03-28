from time import time
from typing import Union

import numpy as np
import pandas as pd

from autopipeline.feature_engineer import SimilarityBase


class Variance(SimilarityBase):

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        self.name = f"variance<={self.threshold}"
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            self._type = "ndarray"
        start = time()
        for col in X.columns.values:
            var = np.var(X[col].values)
            if var <= self.threshold:
                self.to_delete.append(col)
        end = time()
        print("use time:", end - start)
        return self

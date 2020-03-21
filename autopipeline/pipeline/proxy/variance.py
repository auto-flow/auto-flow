from time import time
from typing import Union

import numpy as np
import pandas as pd

from autopipeline.pipeline.proxy.similarity_base import SimilarityBase


class Variance(SimilarityBase):

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        self.name = f"variance<={self.threshold}"
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            self._type = "ndarray"
        start = time()
        col_before = X.shape[1]
        for col in X.columns.values:
            var = np.var(X[col].values)
            if var <= self.threshold:
                self.to_delete.append(col)
        new_X = X.drop(self.to_delete, axis=1)
        col_after = new_X.shape[1]
        end = time()

        print("features before", col_before, ", after", col_after, ",",
              col_before - col_after, f"were deleted by variance<={self.threshold}",
              "use time:", end - start)
        return new_X

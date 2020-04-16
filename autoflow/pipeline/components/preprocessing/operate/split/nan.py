import numpy as np
import pandas as pd
from autoflow.pipeline.components.preprocessing.operate.split.base import BaseSplit

__all__ = ["SplitNan"]


class SplitNan(BaseSplit):
    def calc_R(self, col, rows):
        return np.count_nonzero(pd.isna(col)) / rows

    key1_hp_name = "highR"
    key2_hp_name = "lowR"
    key1_default_name = "highR_nan"
    key2_default_name = "lowR_nan"

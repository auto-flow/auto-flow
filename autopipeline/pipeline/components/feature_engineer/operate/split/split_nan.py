import numpy as np
import pandas as pd
from autopipeline.pipeline.components.feature_engineer.operate.split.base_split import BaseSplit

__all__ = ["SplitNan"]


class SplitNan(BaseSplit):
    def calc_R(self, col, rows):
        return np.count_nonzero(pd.isna(col)) / rows

    key1_hp_name = "highR_nan_name"
    key2_hp_name = "lowR_nan_name"
    key1_default_name = "highR_nan"
    key2_default_name = "lowR_nan"

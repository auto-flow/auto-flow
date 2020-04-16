import numpy as np
import pandas as pd

from autoflow.pipeline.components.preprocessing.operate.split.base import BaseSplit
from autoflow.utils.data import is_cat

__all__ = ["SplitCatNum"]


class SplitCatNum(BaseSplit):
    def calc_R(self, col, rows):
        return np.unique(col).size / rows

    key1_hp_name = "cat_name"
    key2_hp_name = "num_name"
    key1_default_name = "cat"
    key2_default_name = "num"
    key1 = "cat"
    key2 = "num"

    def judge_keyname(self, col: pd.Series, rows):
        if is_cat(col):
            keyname = self.key1
        else:
            keyname = self.key2
        return keyname

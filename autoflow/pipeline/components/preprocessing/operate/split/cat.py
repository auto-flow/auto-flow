import numpy as np

from autoflow.pipeline.components.preprocessing.operate.split.base import BaseSplit

__all__ = ["SplitCat"]


class SplitCat(BaseSplit):
    def calc_R(self, col, rows):
        return np.unique(col.astype("str")).size / rows

    key1_hp_name = "highR"
    key2_hp_name = "lowR"
    key1_default_name = "highR_cat"
    key2_default_name = "lowR_cat"

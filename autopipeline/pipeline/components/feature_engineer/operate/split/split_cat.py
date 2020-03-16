import numpy as np

from autopipeline.pipeline.components.feature_engineer.operate.split.base_split import BaseSplit

__all__ = ["SplitCat"]


class SplitCat(BaseSplit):
    def calc_R(self, col, rows):
        return np.unique(col.astype("str")).size / rows

    key1_hp_name = "highR_cat_name"
    key2_hp_name = "lowR_cat_name"
    key1_default_name = "highR_cat"
    key2_default_name = "lowR_cat"

import numpy as np
import pandas as pd
from autoflow.pipeline.components.preprocessing.operate.split.base import BaseSplit
from autoflow.utils.data import is_highR_nan

__all__ = ["SplitNan"]


class SplitNan(BaseSplit):

    key1_hp_name = "highR"
    key2_hp_name = "lowR"
    key1_default_name = "highR_nan"
    key2_default_name = "lowR_nan"

    def judge_keyname(self, col, rows):
        if is_highR_nan(col, self.threshold):
            keyname = self.key1
        else:
            keyname = self.key2
        return keyname

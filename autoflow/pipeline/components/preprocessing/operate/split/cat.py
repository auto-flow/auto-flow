from autoflow.pipeline.components.preprocessing.operate.split.base import BaseSplit
from autoflow.utils.data import is_highR_cat

__all__ = ["SplitCat"]


class SplitCat(BaseSplit):
    key1_hp_name = "highR"
    key2_hp_name = "lowR"
    key1_default_name = "highR_cat"
    key2_default_name = "lowR_cat"

    def judge_keyname(self, col, rows):
        if is_highR_cat(col, self.threshold):
            keyname = self.key1
        else:
            keyname = self.key2
        return keyname

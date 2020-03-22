from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm
from autopipeline.pipeline.dataframe import GenericDataFrame

__all__ = ["BaseSplit"]


class BaseSplit(AutoPLPreprocessingAlgorithm):
    key1_hp_name = None
    key2_hp_name = None
    key1_default_name = None
    key2_default_name = None
    key1 = "highR"
    key2 = "lowR"
    default_threshold = 0.5

    def judge_keyname(self, col, rows):
        R = self.calc_R(col, rows)
        if R >= self.threshold:
            keyname = self.key1
        else:
            keyname = self.key2
        return keyname

    def calc_R(self, col, rows):
        raise NotImplementedError

    def fit(self, X_train: GenericDataFrame, y_train=None,
            X_valid=None, y_valid=None,
            X_test=None, y_test=None):
        self.threshold = self.hyperparams.get("threshold", self.default_threshold)
        info = {
            self.key1: defaultdict(list),
            self.key2: defaultdict(list),
        }
        X: GenericDataFrame = X_train.filter_feat_grp(self.in_feat_grp)
        rows = X.shape[0]
        for i, (col_name, feat_grp, origin_grp) in enumerate(zip(X.columns, X.feat_grp, X.origin_grp)):
            col = X.iloc[:, i]
            keyname = self.judge_keyname(col, rows)
            info[keyname]["col_name"].append(col_name)
            info[keyname]["feat_grp"].append(feat_grp)
            info[keyname]["origin_grp"].append(origin_grp)
        self.info = info
        return self

    def process(self, X_origin: Optional[GenericDataFrame]) -> Optional[GenericDataFrame]:
        if X_origin is None:
            return None
        X = X_origin.filter_feat_grp(self.in_feat_grp)
        highR = self.hyperparams.get(self.key1_hp_name, self.key1_default_name)
        lowR = self.hyperparams.get(self.key2_hp_name, self.key2_default_name)
        collection = {
            highR: defaultdict(list),
            lowR: defaultdict(list),
        }
        for i, (col_name, feat_grp, origin_grp) in enumerate(zip(X.columns, X.feat_grp, X.origin_grp)):
            col = X.iloc[:, i]
            if col_name in self.info[self.key1]["col_name"]:
                keyname = highR
            else:
                keyname = lowR
            collection[keyname]["X"].append(col)
            collection[keyname]["col_name"].append(col_name)
            collection[keyname]["feat_grp"].append(feat_grp)
            collection[keyname]["origin_grp"].append(origin_grp)
        dfs = []
        for feat_grp_name, dict_ in collection.items():
            X = np.array(dict_["X"]).T
            origin_grp = dict_["origin_grp"]
            feat_grp = [feat_grp_name] * len(origin_grp)
            if X.shape == (0,):
                X = np.zeros([X_origin.shape[0], 0])
            df = GenericDataFrame(pd.DataFrame(X, columns= dict_["col_name"]), feat_grp=feat_grp, origin_grp=origin_grp)
            dfs.append(df)
        assert len(dfs) == 2
        df = dfs[0].concat_two(dfs[0], dfs[1])
        return X_origin.replace_feat_grp(self.in_feat_grp, df, df.feat_grp, df.origin_grp)

    def transform(self, X_train=None, X_valid=None, X_test=None, is_train=False):
        return {
            "X_train": self.process(X_train),
            "X_valid": self.process(X_valid),
            "X_test": self.process(X_test),
        }

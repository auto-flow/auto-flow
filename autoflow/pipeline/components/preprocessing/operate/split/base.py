from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm
from autoflow.pipeline.dataframe import GenericDataFrame

__all__ = ["BaseSplit"]


class BaseSplit(AutoFlowFeatureEngineerAlgorithm):
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
        X: GenericDataFrame = X_train.filter_feature_groups(self.in_feature_groups)
        rows = X.shape[0]
        for i, (col_name, feature_groups, columns_metadata) in enumerate(zip(X.columns, X.feature_groups, X.columns_metadata)):
            col = X.iloc[:, i]
            keyname = self.judge_keyname(col, rows)
            info[keyname]["col_name"].append(col_name)
            info[keyname]["feature_groups"].append(feature_groups)
            info[keyname]["columns_metadata"].append(columns_metadata)
        self.info = info
        return self

    def process(self, X_origin: Optional[GenericDataFrame]) -> Optional[GenericDataFrame]:
        if X_origin is None:
            return None
        X = X_origin.filter_feature_groups(self.in_feature_groups)
        highR = self.hyperparams.get(self.key1_hp_name, self.key1_default_name)
        lowR = self.hyperparams.get(self.key2_hp_name, self.key2_default_name)
        collection = {
            highR: defaultdict(list),
            lowR: defaultdict(list),
        }
        for i, (col_name, feature_groups, columns_metadata) in enumerate(zip(X.columns, X.feature_groups, X.columns_metadata)):
            col = X.iloc[:, i]
            if col_name in self.info[self.key1]["col_name"]:
                keyname = highR
            else:
                keyname = lowR
            collection[keyname]["X"].append(col)
            collection[keyname]["col_name"].append(col_name)
            collection[keyname]["feature_groups"].append(feature_groups)
            collection[keyname]["columns_metadata"].append(columns_metadata)
        dfs = []
        for feature_groups_name, dict_ in collection.items():
            X = np.array(dict_["X"]).T
            columns_metadata = dict_["columns_metadata"]
            feature_groups = [feature_groups_name] * len(columns_metadata)
            if X.shape == (0,):
                X = np.zeros([X_origin.shape[0], 0])
            df = GenericDataFrame(pd.DataFrame(X, columns= dict_["col_name"]), feature_groups=feature_groups, columns_metadata=columns_metadata)
            dfs.append(df)
        assert len(dfs) == 2
        df = dfs[0].concat_two(dfs[0], dfs[1])
        return X_origin.replace_feature_groups(self.in_feature_groups, df, df.feature_groups, df.columns_metadata)

    def transform(self, X_train=None, X_valid=None, X_test=None, y_train=None):
        return {
            "X_train": self.process(X_train),
            "X_valid": self.process(X_valid),
            "X_test": self.process(X_test),
            "y_train": y_train
        }

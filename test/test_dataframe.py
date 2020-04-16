import unittest

import numpy as np
import pandas as pd

from autoflow.pipeline.dataframe import GenericDataFrame


class TestGeneralDataFrame(unittest.TestCase):
    def test_filter_feature_groups(self):
        df = pd.read_csv("../examples/classification/train_classification.csv")
        df2 = GenericDataFrame(df, feature_groups=["id"] + ["num"] * 2 + ["cat"] * 9)
        df3 = df2.filter_feature_groups(["num", "id"])
        self.assertTrue(isinstance(df3, GenericDataFrame))
        self.assertTrue(np.all(df3.columns_metadata == pd.Series(["id", "num", "num"])))
        self.assertTrue(np.all(df3.feature_groups == pd.Series(["id", "num", "num"])))

    def test_replace_feature_groups(self):
        df = pd.read_csv("../examples/classification/train_classification.csv")
        suffix = ["num"] * 2 + ["cat"] * 2 + ["num"] * 5 + ["cat"] * 2
        feature_groups = ["id"] + suffix

        df2 = GenericDataFrame(df, feature_groups=feature_groups)
        # test 1->2
        selected = df2.filter_feature_groups("id").values
        selected = np.hstack([selected, selected])
        df3 = df2.replace_feature_groups("id", selected, "id2")
        self.assertTrue(np.all(df3.feature_groups == pd.Series(suffix + ["id2"] * 2)))
        self.assertTrue(np.all(df3.columns_metadata == pd.Series(suffix + ["id"] * 2)))
        # test 1->1
        selected = df2.filter_feature_groups("id").values
        selected = np.hstack([selected])
        df3 = df2.replace_feature_groups("id", selected, "id2")
        self.assertTrue(np.all(df3.feature_groups == pd.Series(suffix + ["id2"])))
        self.assertTrue(np.all(df3.columns_metadata == pd.Series(suffix + ["id"])))
        # test 1->0
        selected = df2.filter_feature_groups("id").values
        selected = np.zeros([selected.shape[0], 0])
        df3 = df2.replace_feature_groups("id", selected, "id2")
        self.assertTrue(np.all(df3.feature_groups == pd.Series(suffix)))
        self.assertTrue(np.all(df3.columns_metadata == pd.Series(suffix)))


if __name__ == '__main__':
    df = pd.read_csv("../examples/classification/train_classification.csv")
    df2 = GenericDataFrame(df, feature_groups=["id"] + ["num"] * 2 + ["cat"] * 9)
    df3 = df2.filter_feature_groups(["num", "id"])

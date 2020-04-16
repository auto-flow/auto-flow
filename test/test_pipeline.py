import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold

from autoflow import constants
from autoflow.pipeline.components.classification.sgd import SGD
from autoflow.pipeline.components.preprocessing.encode.one_hot import OneHotEncoder
from autoflow.pipeline.components.preprocessing.impute.fill_cat import FillCat
from autoflow.pipeline.components.preprocessing.impute.fill_num import FillNum
from autoflow.pipeline.dataframe import GenericDataFrame
from autoflow.pipeline.pipeline import GenericPipeline
from autoflow.utils.logging import get_logger


class TestPipeline(unittest.TestCase):
    def test_pipeline(self):
        self.logger=get_logger(self)
        df = pd.read_csv("../examples/classification/train_classification.csv")
        y = df.pop("Survived").values
        df = df.loc[:, ["Sex", "Cabin", "Age"]]
        feature_groups = ["cat_nan", "cat_nan", "num_nan"]
        df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=10)
        df_train = GenericDataFrame(df_train, feature_groups=feature_groups)
        df_test = GenericDataFrame(df_test, feature_groups=feature_groups)
        cv = KFold(n_splits=5, random_state=10, shuffle=True)
        train_ix, valid_ix = next(cv.split(df_train))

        df_train, df_valid = df_train.split([train_ix, valid_ix])
        y_valid = y_train[valid_ix]
        y_train = y_train[train_ix]

        fill_cat = FillCat()
        fill_cat.in_feature_groups = "cat_nan"
        fill_cat.out_feature_groups = "cat"
        fill_cat.update_hyperparams({"strategy": "<NULL>"})

        fill_num = FillNum()
        fill_num.in_feature_groups = "num_nan"
        fill_num.out_feature_groups = "num"
        fill_num.update_hyperparams({"strategy": "median"})

        ohe = OneHotEncoder()
        ohe.in_feature_groups = "cat"
        ohe.out_feature_groups = "num"

        sgd = SGD()
        sgd.in_feature_groups = "num"
        sgd.update_hyperparams({"loss": "log", "random_state": 10})

        pipeline = GenericPipeline([
            ("fill_cat", fill_cat),
            ("fill_num", fill_num),
            ("ohe", ohe),
            ("sgd", sgd),
        ])

        pipeline.fit(df_train, y_train, df_valid, y_valid, df_test, y_test)
        pred_train = pipeline.predict(df_train)
        pred_test = pipeline.predict(df_test)
        pred_valid = pipeline.predict(df_valid)
        score_valid = pipeline.predict_proba(df_valid)
        self.logger.info(accuracy_score(y_train, pred_train))
        self.logger.info(accuracy_score(y_valid, pred_valid))
        self.logger.info(accuracy_score(y_test, pred_test))
        result = pipeline.procedure(constants.binary_classification_task, df_train, y_train, df_valid, y_valid, df_test,
                                 y_test)
        pred_test = result["pred_test"]
        pred_valid = result["pred_valid"]
        self.logger.info(accuracy_score(y_valid, (pred_valid > .5).astype("int")[:, 1]))
        self.logger.info(accuracy_score(y_test, (pred_test > .5).astype("int")[:, 1]))

        pipeline = GenericPipeline([
            ("fill_cat", fill_cat),
            ("fill_num", fill_num),
            ("ohe", ohe),
        ])

        pipeline.fit(df_train, y_train, df_valid, y_valid, df_test, y_test)
        ret1 = pipeline.transform(df_train, df_valid, df_test)
        ret2 = pipeline.fit_transform(df_train, y_train, df_valid, y_valid, df_test, y_test)
        for key in ["X_train", "X_valid", "X_test"]:
            assert np.all(ret1[key] == ret2[key])

        pipeline = GenericPipeline([
            ("sgd", sgd),
        ])

        result = pipeline.procedure(constants.binary_classification_task, ret1["X_train"], y_train, ret1["X_valid"],
                                 y_valid,
                                 ret1["X_test"], y_test)
        pred_test = result["pred_test"]
        pred_valid = result["pred_valid"]
        self.logger.info(accuracy_score(y_valid, (pred_valid > .5).astype("int")[:, 1]))
        self.logger.info(accuracy_score(y_test, (pred_test > .5).astype("int")[:, 1]))

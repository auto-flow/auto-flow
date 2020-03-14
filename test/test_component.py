import unittest
import pandas as pd

from autopipeline.pipeline.components.classification.sgd import SGD
from autopipeline.pipeline.components.feature_engineer.encode.one_hot_encode import OneHotEncoder
from autopipeline.pipeline.components.feature_engineer.impute.fill_cat import FillCat
from autopipeline.pipeline.components.feature_engineer.impute.fill_num import FillNum
from autopipeline.pipeline.dataframe import GeneralDataFrame
from sklearn.metrics import accuracy_score

class TestComponent(unittest.TestCase):
    def test_procedure(self):
        df = pd.read_csv("../examples/classification/train_classification.csv")
        y = df.pop("Survived").values
        df = df.loc[:, ["Sex", "Cabin", "Age"]]
        df2 = GeneralDataFrame(df, feat_grp=["cat_nan", "cat_nan", "num_nan"])

        fill_cat = FillCat()
        fill_cat.in_feat_grp = "cat_nan"
        fill_cat.out_feat_grp = "cat"
        fill_cat.update_hyperparams({"strategy": "<NULL>"})

        fill_num = FillNum()
        fill_num.in_feat_grp = "num_nan"
        fill_num.out_feat_grp = "num"
        fill_num.update_hyperparams({"strategy": "median"})

        ohe = OneHotEncoder()
        ohe.in_feat_grp = "cat"
        ohe.out_feat_grp = "num"

        sgd = SGD()
        sgd.in_feat_grp = "num"
        sgd.update_hyperparams({"loss": "log"})

        ret1 = fill_cat.fit_transform(df2)
        ret2 = fill_num.fit_transform(**ret1)
        ret3 = ohe.fit_transform(**ret2)
        sgd.fit(**ret3, y_train=y)

        y_pred = sgd.predict(ret3["X_train"])
        y_score = sgd.predict_proba(ret3["X_train"])
        self.assertEqual(len(y_pred),len(y))
        self.assertGreater(accuracy_score(y,y_pred),0.5)
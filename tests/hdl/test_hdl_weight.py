from sklearn.model_selection import ShuffleSplit

from autoflow import datasets
from autoflow.core.base import AutoFlowEstimator
from autoflow.hdl.hdl2shps import HDL2SHPS
from autoflow.hdl.hdl_constructor import HDL_Constructor
from autoflow.tests.base import LocalResourceTestCase


class TestHDL_Weight(LocalResourceTestCase):
    def test_categorical_hyperparams_weight(self):
        # Make sure you have installed the ConfigSpaceX instead of ConfigSpace.
        # Otherwise, the test case cannot be passed
        df = datasets.load("titanic")
        ss = ShuffleSplit(n_splits=1, random_state=0, test_size=0.25)
        train_ix, test_ix = next(ss.split(df))
        df_train = df.iloc[train_ix, :]
        df_test = df.iloc[test_ix, :]

        hdl_constructor = HDL_Constructor(
            included_classifiers=[{
                "_name": "lightgbm",
                "boosting_type":
                    {"_type": "choice", "_value": [("gbdt", 0.6), "dart", "goss"], "_default": "gbdt"}
            }]
        )
        autoflow_pipeline = AutoFlowEstimator(hdl_constructor=hdl_constructor,
                                              resource_manager=self.mock_resource_manager)
        column_descriptions = {
            "id": "PassengerId",
            "target": "Survived",
            "ignore": "Name"
        }

        autoflow_pipeline.fit(
            X_train=df_train, X_test=df_test, column_descriptions=column_descriptions,
            is_not_realy_run=True
        )
        hdl2shps = HDL2SHPS()
        shps = hdl2shps(autoflow_pipeline.hdl)
        self.assertEqual(
            shps.get_hyperparameter("estimating:lightgbm:boosting_type").probabilities,
            [0.6, 0.2, 0.2]
        )

    def test_algorithms_selection_weight(self):
        # Make sure you have installed the ConfigSpaceX instead of ConfigSpace.
        # Otherwise, the test case cannot be passed
        df = datasets.load("titanic")
        ss = ShuffleSplit(n_splits=1, random_state=0, test_size=0.25)
        train_ix, test_ix = next(ss.split(df))
        df_train = df.iloc[train_ix, :]
        df_test = df.iloc[test_ix, :]
        # fixme: __proba 0.7 0.30004
        hdl_constructor = HDL_Constructor(
            included_classifiers=[
                {
                    "_name": "lightgbm",
                    "boosting_type":
                        {"_type": "choice", "_value": [("gbdt", 0.6), "dart", "goss"], "_default": "gbdt"},
                    "__proba": 0.6
                },
                {
                    "_name": "logistic_regression"
                },
            ]
        )
        autoflow_pipeline = AutoFlowEstimator(hdl_constructor=hdl_constructor,
                                              resource_manager=self.mock_resource_manager)
        column_descriptions = {
            "id": "PassengerId",
            "target": "Survived",
            "ignore": "Name"
        }

        autoflow_pipeline.fit(
            X_train=df_train, X_test=df_test, column_descriptions=column_descriptions,
            is_not_realy_run=True
        )
        hdl2shps = HDL2SHPS()
        shps = hdl2shps(autoflow_pipeline.hdl)
        # print(shps.get_hyperparameter("estimating:__choice__").probabilities)
        self.assertEqual(
            shps.get_hyperparameter("estimating:__choice__").probabilities,
            [0.6, 0.4]
        )

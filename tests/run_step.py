from pathlib import Path

import pandas as pd
from sklearn.model_selection import ShuffleSplit

import autoflow
from autoflow.estimator.base import AutoFlowEstimator
from autoflow.hdl.hdl_constructor import HDL_Constructor
from autoflow.tuner import Tuner


examples_path = Path(autoflow.__file__).parent.parent / "examples"
df = pd.read_csv(examples_path / "data/train_classification.csv")
ss = ShuffleSplit(n_splits=1, random_state=0, test_size=0.25)
train_ix, test_ix = next(ss.split(df))
df_train = df.iloc[train_ix, :]
df_test = df.iloc[test_ix, :]

hdl_constructors = [
    HDL_Constructor(
        DAG_workflow={
            "nan->{highR=highR_nan,lowR=lowR_nan}": "operate.split.nan",
            "lowR_nan->nan": "impute.fill_abnormal",
            "highR_nan->nan": "operate.drop",
            "all->{cat_name=cat,num_name=num}": "operate.split.cat_num",
            "cat->num": "encode.ordinal",
            "num->num": [
                {"_name": "select.from_model_clf", "_select_percent": 80},
                {"_name": "select.rfe_clf", "_select_percent": 80},
            ],
            "num->target": {"_name": "logistic_regression", "_vanilla": True}
        }
    ),
    HDL_Constructor(
        DAG_workflow={
            "nan->{highR=highR_nan,lowR=lowR_nan}": "operate.split.nan",
            "lowR_nan->nan": "impute.fill_abnormal",
            "highR_nan->nan": "operate.drop",
            "all->{cat_name=cat,num_name=num}": "operate.split.cat_num",
            "cat->num": "encode.ordinal",
            "num->num": {"_name": "<mask>",
                         "_select_percent": {"_type": "quniform", "_value": [1, 100, 0.5],
                                                       "_default": 80}},
            "num->target": {"_name": "logistic_regression", "_vanilla": True}
        }
    ),
]

tuners = [
    Tuner(
        run_limit=-1,
        search_method="grid",
        n_jobs=3,
        debug=True
    ),
    Tuner(
        run_limit=50,
        initial_runs=10,
        search_method="smac",
        n_jobs=3,
        debug=True
    ),
]
autoflow_pipeline = AutoFlowEstimator(tuners, hdl_constructors)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}

autoflow_pipeline.fit(
    X_train=df_train, X_test=df_test, column_descriptions=column_descriptions
)

import pandas as pd

from autoflow.estimator.base import AutoFlowEstimator
from autoflow.hdl.hdl_constructor import HDL_Constructor
from autoflow.tuner.tuner import Tuner

df_train = pd.read_csv("./data/train_classification.csv")

hdl_constructors = [
    HDL_Constructor(
        DAG_workflow={
            "nan->{highR=highR_nan,lowR=lowR_nan}": "operate.split.nan",
            "lowR_nan->nan": "impute.fill_abnormal",
            "highR_nan->nan": "operate.drop",
            "all->{cat_name=cat,num_name=num}": "operate.split.cat_num",
            "cat->num": "encode.label",
            "num->selected": [
                {"_name": "select.from_model_clf", "_select_percent": 80},
                {"_name": "select.rfe_clf", "_select_percent": 80},
            ],
            "selected->target": {"_name": "logistic_regression", "_vanilla": True}
        }
    ),
    HDL_Constructor(
        DAG_workflow={
            "nan->{highR=highR_nan,lowR=lowR_nan}": "operate.split.nan",
            "lowR_nan->nan": "impute.fill_abnormal",
            "highR_nan->nan": "operate.drop",
            "all->{cat_name=cat,num_name=num}": "operate.split.cat_num",
            "cat->num": "encode.label",
            "num->selected": {"_name": "<mask>",
                         "_select_percent": {"_type": "quniform", "_value": [1, 100, 0.5],
                                             "_default": 80}},
            "selected->target": {"_name": "logistic_regression", "_vanilla": True}
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
    X_train=df_train, column_descriptions=column_descriptions
)

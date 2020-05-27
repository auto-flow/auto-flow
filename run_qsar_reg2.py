#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from pathlib import Path

import joblib
from sklearn.model_selection import KFold

import autoflow
from autoflow import AutoFlowRegressor, HDL_Constructor
from autoflow.tuner import Tuner

examples_path = Path(autoflow.__file__).parent.parent / "data"
train_df = joblib.load(examples_path / "2198.bz2")
# train_df=load("qsar")

estimators = ["adaboost", "bayesian_ridge", "catboost", "decision_tree", "elasticnet", "extra_trees",
              "gradient_boosting", "k_nearest_neighbors", "lightgbm", "random_forest"]

hdl_constructor = HDL_Constructor(
    DAG_workflow={
        "num->selected": {
            "_name": "select.from_model_reg",
            "_vanilla": True,
            "estimator": "sklearn.ensemble.ExtraTreesRegressor",
            "n_estimators": 10,
            "max_depth": 7,
            "min_samples_split": 10,
            "min_samples_leaf": 10,
            "random_state": 42,
            "n_jobs": 1,
            "_select_percent": {"_type": "quniform", "_value": [1, 100, 1], "_default": 40}
        },
        "selected->target": estimators
    }
)

hdl_constructors = [hdl_constructor]

tuners = [
    # Tuner(
    #     search_method="beam",
    #     search_method_params={
    #         "beam_steps": {
    #             "estimating": estimators,
    #             "num->selected.*_select_percent": [i for i in range(10, 100, 10)]
    #         }
    #     },
    #     debug=True
    # ),
    Tuner(
        search_method="smac",
        run_limit=100,
        initial_runs=1,
        debug=False,
        per_run_time_limit=1200,
        per_run_memory_limit=30 * 1024,
        n_jobs_in_algorithm=-1
    )
]

trained_pipeline = AutoFlowRegressor(
    consider_ordinal_as_cat=False,
    tuner=tuners,
    hdl_constructor=hdl_constructors,
    random_state=42
)
column_descriptions = {
    "id": "Name",
    "ignore": ["Smiles", "labels"],
    "target": "pIC50"
}
# column_descriptions = {
#     "target": "target"
# }
# if not os.path.exists("autoflow_classification.bz2"):
trained_pipeline.fit(
    X_train=train_df, column_descriptions=column_descriptions,
    splitter=KFold(n_splits=5, shuffle=True, random_state=0),
    fit_ensemble_params=False,
    # is_not_realy_run=True
)
print(trained_pipeline)

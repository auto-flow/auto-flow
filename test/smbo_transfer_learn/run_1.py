#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import ShuffleSplit

import autoflow
from autoflow import AutoFlowClassifier

examples_path = Path(autoflow.__file__).parent.parent / "examples"
train_df = pd.read_csv(examples_path / "data/train_classification.csv")
test_df = pd.read_csv(examples_path / "data/test_classification.csv")
trained_pipeline = AutoFlowClassifier(
    initial_runs=8, run_limit=5, n_jobs=1,
    included_classifiers=[
        "libsvm_svc", "logistic_regression",
        "gaussian_nb", "k_nearest_neighbors", "liblinear_svc","lightgbm"],
    debug=True,
    should_store_intermediate_result=True,
    db_type="postgresql", db_params={
        "user": "tqc",
        "host": "0.0.0.0",
        "port": 5432
    }
)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}

trained_pipeline.fit(
    X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
    splitter=ShuffleSplit(n_splits=1, test_size=0.25, random_state=42), fit_ensemble_params=False
)

joblib.dump(trained_pipeline, "autoflow_classification.bz2")
predict_pipeline = joblib.load("autoflow_classification.bz2")
result = predict_pipeline.predict(test_df)
print(result)

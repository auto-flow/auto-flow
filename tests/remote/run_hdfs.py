#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import KFold

import autoflow
from autoflow import AutoFlowClassifier

examples_path = Path(autoflow.__file__).parent.parent / "examples"
train_df = pd.read_csv(examples_path / "data/train_classification.csv")
test_df = pd.read_csv(examples_path / "data/test_classification.csv")
trained_pipeline = AutoFlowClassifier(
    initial_runs=5, run_limit=10, n_jobs=3,
    included_classifiers=["lightgbm"],
    db_type="postgresql",
    db_params={
        "user": "tqc",
        "host": "0.0.0.0",
        "port": 5432
    },
    store_path="/autoflow",
    file_system="hdfs",
    should_store_intermediate_result=True,
    file_system_params={
        "url": "http://0.0.0.0:50070",
        "user": "tqc"
    }
)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}
# if not os.path.exists("autoflow_classification.bz2"):
trained_pipeline.fit(
    X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
    splitter=KFold(n_splits=3, shuffle=True, random_state=42),
    # fit_ensemble_params=False
)
joblib.dump(trained_pipeline, "autoflow_classification.bz2")
predict_pipeline = joblib.load("autoflow_classification.bz2")
result = predict_pipeline.predict(test_df)
print(result)

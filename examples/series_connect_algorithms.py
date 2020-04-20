#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

import joblib
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from autoflow import AutoFlowClassifier

train_df = pd.read_csv("./data/train_classification.csv")
test_df = pd.read_csv("./data/test_classification.csv")
trained_pipeline = AutoFlowClassifier(
    initial_runs=12, run_limit=12, n_jobs=3,
    included_classifiers=[
        "scale.standardize|libsvm_svc", "scale.standardize|k_nearest_neighbors", "scale.standardize|logistic_regression",
        "gaussian_nb", "extra_trees", "lightgbm"
    ],
)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}
trained_pipeline.fit(
    X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
    fit_ensemble_params=False,
    splitter=ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
)
joblib.dump(trained_pipeline, "autoflow_classification.bz2")
predict_pipeline = joblib.load("autoflow_classification.bz2")
result = predict_pipeline.predict(test_df)
print(result)

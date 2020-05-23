#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

import autoflow
from autoflow import AutoFlowClassifier

examples_path = Path(autoflow.__file__).parent.parent / "examples"
train_df = pd.read_csv(examples_path / "data/train_classification.csv")
test_df = pd.read_csv(examples_path / "data/test_classification.csv")
trained_pipeline = AutoFlowClassifier(
    initial_runs=1, run_limit=3, n_jobs=1,
    included_classifiers=["catboost"], debug=True,
    n_jobs_in_algorithm=5
    # should_store_intermediate_result=True,  # 测试对中间结果存储的正确性
)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "text": "Name"
}
# if not os.path.exists("autoflow_classification.bz2"):
trained_pipeline.fit(
    X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
    splitter=KFold(n_splits=3, shuffle=True, random_state=42),
    fit_ensemble_params=True
)
Path("autoflow_classification.pkl").write_bytes(trained_pipeline.pickle())
predict_pipeline = pickle.loads(Path("autoflow_classification.pkl").read_bytes())
test_df = pd.read_csv(examples_path / "data/test_classification.csv")
result = predict_pipeline.predict(test_df)
print(result)

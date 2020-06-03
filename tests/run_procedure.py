#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import pickle
from pathlib import Path

from sklearn.model_selection import KFold

from autoflow import AutoFlowClassifier
from autoflow import datasets

train_df, test_df = datasets.load("titanic", return_train_test=True)
trained_pipeline = AutoFlowClassifier(
    initial_runs=1, run_limit=3, n_jobs=1,
    included_classifiers=["catboost"],
    debug=True,
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
    fit_ensemble_params=True,
    # is_not_realy_run=True
)
Path("autoflow_classification.pkl").write_bytes(trained_pipeline.pickle())
predict_pipeline = pickle.loads(Path("autoflow_classification.pkl").read_bytes())
_, test_df = datasets.load("titanic", return_train_test=True)
result = predict_pipeline.predict(test_df)
print(result)

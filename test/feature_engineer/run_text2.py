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
# 通过各种限制搜索空间，使得网格搜索空间范围为5
trained_pipeline = AutoFlowClassifier(
    initial_runs=1, run_limit=-1, n_jobs=1,
    included_classifiers={"_name": "lightgbm", "_vanilla": True},
    included_highR_nan_imputers="operate.drop",
    included_cat_encoders="encode.ordinal",
    included_highR_cat_encoders="encode.ordinal",
    included_nan_imputers={"_name": "impute.adaptive_fill", "num_strategy": "mean"},
    num2purified_workflow={
        "num->scaled": "scale.standardize",
        "scaled->purified": "operate.keep_going",
    },
    text2purified_workflow={
        "text->tokenized": "text.tokenize.simple",
        "tokenized->purified": [
            {"_name": "text.topic.lda", "num_topics": 16},
            {"_name": "text.topic.lsi", "num_topics": 16},
            {"_name": "text.topic.nmf", "num_topics": 16},
            {"_name": "text.topic.rp", "num_topics": 16},
            {"_name": "text.topic.tsvd", "num_topics": 16},
        ]
    },
    debug=True,
    should_store_intermediate_result=True,
    search_method="grid"
)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "text": "Name"
}
# if not os.path.exists("autoflow_classification.bz2"):
trained_pipeline.fit(
    X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
    splitter=KFold(n_splits=3, shuffle=True, random_state=42)
)
joblib.dump(trained_pipeline, "autoflow_classification.bz2")
predict_pipeline = joblib.load("autoflow_classification.bz2")
result = predict_pipeline.predict(test_df)
print(result)

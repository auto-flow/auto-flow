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
from autoflow.datasets import load

train_df = load("qsar")
trained_pipeline = AutoFlowClassifier(
    initial_runs=1, run_limit=5, n_jobs=1,
    included_classifiers=["lightgbm"], debug=True,
    num2purified_workflow={
        "num->compressed": {"_name": "compress.f1score", "threshold": 0.9, "n_jobs": 12,
                            # "store_intermediate":False
                            },
        "compressed->purified": ["scale.standardize", "operate.merge"],
    }
    # should_store_intermediate_result=True,  # 测试对中间结果存储的正确性
)
column_descriptions = {
    "target": "target"
}
# if not os.path.exists("autoflow_classification.bz2"):
trained_pipeline.fit(
    X_train=train_df,  column_descriptions=column_descriptions,
    splitter=KFold(n_splits=3, shuffle=True, random_state=42),
    fit_ensemble_params=False
)
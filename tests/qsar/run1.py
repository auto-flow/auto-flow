#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import pickle
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import KFold

import autoflow
from autoflow import AutoFlowClassifier

examples_path = Path(autoflow.__file__).parent.parent / "data"
# train_df = joblib.load(examples_path / "2198.bz2")

trained_pipeline = AutoFlowClassifier(
    initial_runs=1, run_limit=3, n_jobs=1,
    included_classifiers=["catboost"], debug=True,
    n_jobs_in_algorithm=5,
    consider_ordinal_as_cat=2
)
column_descriptions = {
    "id": "Name",
    "ignore": ["Smiles", "pIC50"],
    "target": "labels"
}
# if not os.path.exists("autoflow_classification.bz2"):
trained_pipeline.fit(
    X_train="b4e678b56064b1d5d3b60b272fa3deef", column_descriptions=column_descriptions,
    splitter=KFold(n_splits=3, shuffle=True, random_state=42),
    fit_ensemble_params=True,
    is_not_realy_run=True
)
print(trained_pipeline)

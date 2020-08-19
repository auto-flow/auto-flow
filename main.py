#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import os

import numpy as np
from joblib import load

from autoflow.core.classifier import AutoFlowClassifier

X_train, y_train, X_test, y_test, cat = load(os.environ["DATAPATH"])
cat = np.array(cat)
sampled_num_col = X_train.columns[~cat][0]
sampled_cat_col = X_test.columns[cat][0]
rng = np.random.RandomState(42)
mask1 = rng.choice(X_test.index, 20, False)
mask2 = rng.choice(X_test.index, 15, False)
X_test.loc[mask1, sampled_cat_col] = np.nan
X_test.loc[mask2, sampled_num_col] = np.nan
column_descritions = {"cat": X_train.columns[cat].tolist()}
n_workers = int(os.getenv("N_WORKERS", 6))
n_iterations = int(os.getenv("N_ITERATIONS", 6))
min_points_in_model = int(os.getenv("MIN_POINTS_IN_MODEL", 50))

pipe = AutoFlowClassifier(
    imbalance_threshold=1,
    should_record_workflow_step=False,
    db_type="postgresql",
    db_params={
        "user": "tqc",
        "host": "0.0.0.0",
        "port": 5432
    },
    search_record_db_name="autoflow_test",
    config_generator="ET",
    config_generator_params={
        # "acq_func": "EI",
        # "xi": 0,
        # "loss_transformer":None,
        # "bw_method": "scott",
        # "n_samples": 5000,
        "min_points_in_model": min_points_in_model,
        "use_local_search": False,
        "use_thompson_sampling": False,
        # "kde_sample_weight_scaler": None
    },
    n_folds=3,
    warm_start=False,
    random_state=0,
    min_n_samples_for_SH=50,
    concurrent_type="process",
    n_workers=n_workers,
    SH_only=True,
    min_budget=4,
    max_budget=4,
    n_iterations=n_iterations,
    debug_evaluator=True,
    initial_points=None
)
pipe.fit(
    X_train, y_train, X_test, y_test,
    column_descriptions=column_descritions,
    # is_not_realy_run=True,
    fit_ensemble_params=False)
# score = accuracy_score(y_test, y_pred)
score = pipe.score(X_test, y_test)
print(score)

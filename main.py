#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import os
import numpy as np
from joblib import load

from autoflow.core.classifier import AutoFlowClassifier

X_train, y_train, X_test, y_test, cat = load(os.environ["DATAPATH"])
cat=np.array(cat)
column_descritions={"cat": X_train.columns[cat].tolist()}
n_workers = int(os.getenv("N_WORKERS", 1))
n_iterations = int(os.getenv("N_ITERATIONS", 100))
min_points_in_model = int(os.getenv("MIN_POINTS_IN_MODEL", 30))

pipe = AutoFlowClassifier(
    # DAG_workflow={
    #     "num->target": [
    #         "lightgbm"
    #     ]
    # },
    config_generator="ET",
    config_generator_params={
        # "acq_func": "EI",
        # "xi": 0,
        # "loss_transformer":None,
        # "bw_method": "scott",
        # "n_samples": 5000,
        "min_points_in_model": min_points_in_model,
        "use_local_search": False,
        # "use_thompson_sampling":False,
        # "kde_sample_weight_scaler": None
    },
    n_folds=5,
    warm_start=False,
    random_state=0,
    min_n_samples_for_SH=50,
    concurrent_type="process",
    n_jobs_in_algorithm=3,
    n_workers=n_workers,
    SH_only=True,
    min_budget=4,
    max_budget=4,
    n_iterations=n_iterations,
    debug_evaluator=True,
)
pipe.fit(
    X_train, y_train, X_test, y_test,
    column_descriptions=column_descritions,
    # is_not_realy_run=True,
    fit_ensemble_params=False)
# score = accuracy_score(y_test, y_pred)
score = pipe.score(X_test, y_test)
print(score)

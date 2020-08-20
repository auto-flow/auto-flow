#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import os
from collections import Counter

import numpy as np
from joblib import load

from autoflow.core.classifier import AutoFlowClassifier

n_workers = int(os.getenv("N_WORKERS", 1))
n_iterations = int(os.getenv("N_ITERATIONS", 100))
min_points_in_model = int(os.getenv("MIN_POINTS_IN_MODEL", 50))

initial_points = [
    {'estimating:__choice__': 'gbt_lr', 'preprocessing:cat->normed:__choice__': 'encode.one_hot',
     'preprocessing:combined->normed:__choice__': 'encode.cat_boost',
     'preprocessing:highC_cat->combined:__choice__': 'encode.combine_rare',
     'preprocessing:impute:__choice__': 'impute.gbt', 'preprocessing:impute:missing_rate': 0.6000000000000001,
     'preprocessing:normed->final:__choice__': 'generate.autofeat',
     'preprocessing:num->normed:__choice__': 'operate.keep_going',
     'process_sequence': 'impute;cat->normed;highC_cat->combined;combined->normed;num->normed;normed->final',
     'strategies:balance:__choice__': 'None', 'estimating:gbt_lr:C': 0.013036445394312687,
     'estimating:gbt_lr:bagging_fraction': 0.65, 'estimating:gbt_lr:bagging_freq': 4,
     'estimating:gbt_lr:boosting_type': 'gbdt', 'estimating:gbt_lr:early_stopping_rounds': '32:int',
     'estimating:gbt_lr:feature_fraction': 0.65, 'estimating:gbt_lr:lambda_l1': 0.008792723816659204,
     'estimating:gbt_lr:lambda_l2': 0.0002692462154729917, 'estimating:gbt_lr:learning_rate': 0.015005684851667521,
     'estimating:gbt_lr:max_depth': 30, 'estimating:gbt_lr:min_child_weight': 0.0036284344828391786,
     'estimating:gbt_lr:n_estimators': '256:int', 'estimating:gbt_lr:n_jobs': '12:int',
     'estimating:gbt_lr:num_leaves': 90, 'estimating:gbt_lr:random_state': '42:int',
     'estimating:gbt_lr:subsample_for_bin': 180000, 'estimating:gbt_lr:use_categorical_feature': 'False:bool',
     'preprocessing:cat->normed:encode.one_hot:placeholder': 'placeholder',
     'preprocessing:combined->normed:encode.cat_boost:placeholder': 'placeholder',
     'preprocessing:highC_cat->combined:encode.combine_rare:copy': 'False:bool',
     'preprocessing:highC_cat->combined:encode.combine_rare:minimum_fraction': '0.1:float',
     'preprocessing:impute:impute.gbt:copy': 'False:bool', 'preprocessing:impute:impute.gbt:n_jobs': '12:int',
     'preprocessing:impute:impute.gbt:random_state': '42:int',
     'preprocessing:normed->final:generate.autofeat:n_jobs': '12:int',
     'preprocessing:normed->final:generate.autofeat:random_state': '42:int',
     'preprocessing:normed->final:generate.autofeat:sqr_op': 'False:bool',
     'preprocessing:num->normed:operate.keep_going:placeholder': 'placeholder',
     'strategies:balance:None:placeholder': 'placeholder'}
]

X_train, y_train, cat = load(os.environ["DATAPATH"])
cat = np.array(cat)
column_descritions = {"cat": X_train.columns[cat].tolist()}
counter = Counter(y_train)
minors = []
for label, count in counter.items():
    if count <= 20:
        minors.append(label)

# mask = ~pd.Series(y_train).isin(minors)
# y_train = y_train[mask]
# X_train = X_train.iloc[mask, :]

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
    initial_points=initial_points
)
pipe.fit(
    X_train, y_train,  # X_test, y_test,
    column_descriptions=column_descritions,
    # is_not_realy_run=True,
    fit_ensemble_params=False
)
# score = accuracy_score(y_test, y_pred)
# score = pipe.score(X_test, y_test)
# print(score)

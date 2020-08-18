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
n_workers = int(os.getenv("N_WORKERS", 1))
n_iterations = int(os.getenv("N_ITERATIONS", 100))
min_points_in_model = int(os.getenv("MIN_POINTS_IN_MODEL", 50))

initial_points = [
    {'estimating:__choice__': 'tabular_nn',
     'preprocessing:cat->normed:__choice__': 'encode.one_hot',
     'preprocessing:combined->normed:__choice__': 'encode.cat_boost',
     'preprocessing:highC_cat->combined:__choice__': 'encode.combine_rare',
     'preprocessing:impute:__choice__': 'impute.simple',
     'preprocessing:normed->final:__choice__': 'generate.autofeat',
     'preprocessing:num->normed:__choice__': 'operate.keep_going',
     'process_sequence': 'impute;cat->normed;highC_cat->combined;combined->normed;num->normed;normed->final',
     'estimating:tabular_nn:af_hidden': 'leaky_relu',
     'estimating:tabular_nn:af_output': 'linear',
     'estimating:tabular_nn:batch_size': '1024:int',
     'estimating:tabular_nn:class_weight': 'None:NoneType',
     'estimating:tabular_nn:dropout_hidden': 0.45,
     'estimating:tabular_nn:dropout_output': 0.1,
     'estimating:tabular_nn:early_stopping_rounds': '16:int',
     'estimating:tabular_nn:early_stopping_tol': '0:int',
     'estimating:tabular_nn:layer1': 256,
     'estimating:tabular_nn:layer2': 160,
     'estimating:tabular_nn:lr': '0.01:float',
     'estimating:tabular_nn:max_epoch': '128:int',
     'estimating:tabular_nn:max_layer_width': '2056:int',
     'estimating:tabular_nn:min_layer_width': '32:int',
     'estimating:tabular_nn:n_jobs': '12:int',
     'estimating:tabular_nn:optimizer': 'adam',
     'estimating:tabular_nn:random_state': '42:int',
     'estimating:tabular_nn:use_bn': 'False:bool',
     'estimating:tabular_nn:verbose': '-1:int',
     'preprocessing:cat->normed:encode.one_hot:placeholder': 'placeholder',
     'preprocessing:combined->normed:encode.cat_boost:placeholder': 'placeholder',
     'preprocessing:highC_cat->combined:encode.combine_rare:copy': 'False:bool',
     'preprocessing:highC_cat->combined:encode.combine_rare:minimum_fraction': '0.1:float',
     'preprocessing:impute:impute.simple:cat_strategy': 'most_frequent',
     'preprocessing:impute:impute.simple:copy': 'False:bool',
     'preprocessing:impute:impute.simple:num_strategy': 'mean',
     'preprocessing:normed->final:generate.autofeat:n_jobs': '12:int',
     'preprocessing:normed->final:generate.autofeat:random_state': '42:int',
     'preprocessing:normed->final:generate.autofeat:sqr_op': 'False:bool',
     'preprocessing:num->normed:operate.keep_going:placeholder': 'placeholder'},
    {
        'estimating:__choice__': 'tabular_nn',
        'preprocessing:cat->normed:__choice__': 'encode.one_hot',
        'preprocessing:combined->normed:__choice__': 'encode.cat_boost',
        'preprocessing:highC_cat->combined:__choice__': 'encode.combine_rare',
        'preprocessing:impute:__choice__': 'impute.simple',
        'preprocessing:normed->final:__choice__': 'generate.autofeat',
        'preprocessing:num->normed:__choice__': 'scale.standard',
        'process_sequence': 'impute;cat->normed;highC_cat->combined;combined->normed;num->normed;normed->final',
        'estimating:tabular_nn:af_hidden': 'leaky_relu',
        'estimating:tabular_nn:af_output': 'linear',
        'estimating:tabular_nn:batch_size': '1024:int',
        'estimating:tabular_nn:class_weight': 'None:NoneType',
        'estimating:tabular_nn:dropout_hidden': 0.4,
        'estimating:tabular_nn:dropout_output': 0.35000000000000003,
        'estimating:tabular_nn:early_stopping_rounds': '16:int',
        'estimating:tabular_nn:early_stopping_tol': '0:int',
        'estimating:tabular_nn:layer1': 256,
        'estimating:tabular_nn:layer2': 160,
        'estimating:tabular_nn:lr': '0.01:float',
        'estimating:tabular_nn:max_epoch': '128:int',
        'estimating:tabular_nn:max_layer_width': '2056:int',
        'estimating:tabular_nn:min_layer_width': '32:int',
        'estimating:tabular_nn:n_jobs': '12:int',
        'estimating:tabular_nn:optimizer': 'adam',
        'estimating:tabular_nn:random_state': '42:int',
        'estimating:tabular_nn:use_bn': 'False:bool',
        'estimating:tabular_nn:verbose': '-1:int',
        'preprocessing:cat->normed:encode.one_hot:placeholder': 'placeholder',
        'preprocessing:combined->normed:encode.cat_boost:placeholder': 'placeholder',
        'preprocessing:highC_cat->combined:encode.combine_rare:copy': 'False:bool',
        'preprocessing:highC_cat->combined:encode.combine_rare:minimum_fraction': '0.001:float',
        'preprocessing:impute:impute.simple:cat_strategy': 'constant',
        'preprocessing:impute:impute.simple:copy': 'False:bool',
        'preprocessing:impute:impute.simple:num_strategy': 'median',
        'preprocessing:normed->final:generate.autofeat:n_jobs': '12:int',
        'preprocessing:normed->final:generate.autofeat:random_state': '42:int',
        'preprocessing:normed->final:generate.autofeat:sqr_op': 'True:bool',
        'preprocessing:num->normed:scale.standard:placeholder': 'placeholder'
    }
]

pipe = AutoFlowClassifier(
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
    X_train, y_train, X_test, y_test,
    column_descriptions=column_descritions,
    # is_not_realy_run=True,
    fit_ensemble_params=False)
# score = accuracy_score(y_test, y_pred)
score = pipe.score(X_test, y_test)
print(score)

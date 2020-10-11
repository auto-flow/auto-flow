#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow import AutoFlowClassifier
from autoflow.datasets import load

df = load("titanic")
db_params = {
    "user": "postgres",
    "host": "127.0.0.1",
    "port": 5432,
    "password": "123"
}
initial_points=[{
    "estimating:__choice__": "lightgbm",
    "preprocessing:cat->normed:__choice__": "encode.ordinal",
    "preprocessing:combined->normed:__choice__": "encode.entity",
    "preprocessing:highC_cat->combined:__choice__": "encode.combine_rare",
    "preprocessing:impute:__choice__": "impute.gbt",
    "preprocessing:impute:missing_rate": 1.0,
    "preprocessing:normed->final:__choice__": "generate.autofeat",
    "preprocessing:num->normed:__choice__": "scale.standard",
    "process_sequence": "impute;cat->normed;highC_cat->combined;combined->normed;num->normed;normed->final",
    "estimating:lightgbm:bagging_fraction": 0.95,
    "estimating:lightgbm:bagging_freq": 8,
    "estimating:lightgbm:boosting_type": "gbdt",
    "estimating:lightgbm:early_stopping_rounds": "256:int",
    "estimating:lightgbm:feature_fraction": 1.0,
    "estimating:lightgbm:lambda_l1": 6.856983598741903e-07,
    "estimating:lightgbm:lambda_l2": 0.0006789453682348903,
    "estimating:lightgbm:learning_rate": 0.2,
    "estimating:lightgbm:max_depth": 55,
    "estimating:lightgbm:min_child_weight": 1e-07,
    "estimating:lightgbm:n_estimators": "2048:int",
    "estimating:lightgbm:n_jobs": "12:int",
    "estimating:lightgbm:num_leaves": 150,
    "estimating:lightgbm:random_state": "42:int",
    "estimating:lightgbm:subsample_for_bin": 20000,
    "estimating:lightgbm:use_categorical_feature": "False:bool",
    "preprocessing:cat->normed:encode.ordinal:placeholder": "placeholder",
    "preprocessing:combined->normed:encode.entity:copy": "False:bool",
    "preprocessing:combined->normed:encode.entity:max_epoch": "20:int",
    "preprocessing:combined->normed:encode.entity:n_jobs": "12:int",
    "preprocessing:combined->normed:encode.entity:random_state": "42:int",
    "preprocessing:highC_cat->combined:encode.combine_rare:copy": "False:bool",
    "preprocessing:highC_cat->combined:encode.combine_rare:minimum_fraction": "0.001:float",
    "preprocessing:impute:impute.gbt:copy": "False:bool",
    "preprocessing:impute:impute.gbt:n_jobs": "12:int",
    "preprocessing:impute:impute.gbt:random_state": "42:int",
    "preprocessing:normed->final:generate.autofeat:n_jobs": "12:int",
    "preprocessing:normed->final:generate.autofeat:random_state": "42:int",
    "preprocessing:normed->final:generate.autofeat:sqr_op": "True:bool",
    "preprocessing:num->normed:scale.standard:placeholder": "placeholder"
}]
search_record_db_name = "autoflow_meta_bo"
autoflow = AutoFlowClassifier(
    store_path=f"~/{search_record_db_name}",
    evaluation_strategy="3CV",
    # concurrent_type="thread",
    concurrent_type="process",
    db_params=db_params,
    db_type="postgresql",
    search_record_db_name=search_record_db_name,
    config_generator="et-based-ambo",
    config_generator_params=dict(
        use_thompson_sampling=2,
        min_points_in_model=15
    ),
    n_iterations=1,
    mtl_path=f"~/{search_record_db_name}/metalearning",
    # included_classifier=["lightgbm"],
    n_workers=1,
    initial_points=initial_points,
    use_metalearning=False,
    warm_start=False,
    per_run_time_limit=4
)
autoflow.fit(X_train=df, column_descriptions={"target": "Survived"}, fit_ensemble_params=None)



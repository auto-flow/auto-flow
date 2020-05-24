#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
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
    n_jobs_in_algorithm=5, search_method="beam",
    search_method_params={
        "beam_steps": {
            "nan->imputed.*num_strategy": ["median", "mean"],
            "tokenized->purified": ['text.topic.tsvd', "text.topic.lsi", "text.topic.nmf"],
            "tokenized->purified.*num_topics": [5, 8, 16, 32],
        }
    }
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
# print(trained_pipeline)
# Path("autoflow_classification.pkl").write_bytes(trained_pipeline.pickle())
# predict_pipeline = pickle.loads(Path("autoflow_classification.pkl").read_bytes())
# test_df = pd.read_csv(examples_path / "data/test_classification.csv")
# result = predict_pipeline.predict(test_df)
# print(result)

# {'highR_nan->nan': ('operate.drop', 'operate.keep_going'),
#  'nan->imputed': ('impute.adaptive_fill',),
#  'imputed->cat,num': [{'_name': 'operate.split',
#    'column2fg': "{'Age': 'num', 'Cabin': 'cat', 'Embarked': 'cat', 'Fare': 'num'}:dict"}],
#  'cat->purified': ('encode.one_hot', 'encode.ordinal', 'encode.cat_boost'),
#  'highR_cat->purified': ('operate.drop', 'encode.ordinal', 'encode.cat_boost'),
#  'text->tokenized': ['text.tokenize.simple'],
#  'tokenized->purified': ['text.topic.tsvd',
#   'text.topic.lsi',
#   'text.topic.nmf'],
#  'num->scaled': ['scale.standardize', 'operate.keep_going'],
#  'scaled->purified': ['operate.keep_going', 'transform.power'],
#  'purified->final': ['operate.keep_going'],
#  'final->target': ['catboost']}

# Configuration space object:
#   Hyperparameters:
#     estimating:__choice__, Type: Categorical, Choices: {catboost}, Default: catboost
#     estimating:catboost:border_count, Type: UniformInteger, Range: [1, 1000], Default: 32, on log-scale
#     estimating:catboost:early_stopping_rounds, Type: Constant, Value: 250:int
#     estimating:catboost:l2_leaf_reg, Type: UniformFloat, Range: [0.1, 100.0], Default: 3.0, on log-scale
#     estimating:catboost:learning_rate, Type: UniformFloat, Range: [0.01, 0.2], Default: 0.1, on log-scale
#     estimating:catboost:max_depth, Type: UniformInteger, Range: [1, 15], Default: 7, Q: 1
#     estimating:catboost:n_estimators, Type: Constant, Value: 5000:int
#     estimating:catboost:n_jobs, Type: Constant, Value: 1:int
#     estimating:catboost:random_state, Type: Constant, Value: 42:int
#     estimating:catboost:subsample, Type: UniformFloat, Range: [0.1, 1.0], Default: 1.0, Q: 0.1
#     estimating:catboost:use_best_model, Type: Constant, Value: True:bool
#     preprocessing:00highR_nan->nan:__choice__, Type: Categorical, Choices: {operate.drop, operate.keep_going}, Default: operate.drop
#     preprocessing:00highR_nan->nan:operate.drop:placeholder, Type: Constant, Value: placeholder
#     preprocessing:00highR_nan->nan:operate.keep_going:placeholder, Type: Constant, Value: placeholder
#     preprocessing:01nan->imputed:__choice__, Type: Categorical, Choices: {impute.adaptive_fill}, Default: impute.adaptive_fill
#     preprocessing:01nan->imputed:impute.adaptive_fill:cat_strategy, Type: Constant, Value: most_frequent
#     preprocessing:01nan->imputed:impute.adaptive_fill:num_strategy, Type: Categorical, Choices: {median, mean}, Default: median
#     preprocessing:02imputed->cat,num:__choice__, Type: Categorical, Choices: {operate.split}, Default: operate.split
#     preprocessing:02imputed->cat,num:operate.split:column2fg, Type: Constant, Value: {'Age': 'num', 'Cabin': 'cat', 'Embarked': 'cat', 'Fare': 'num'}:dict
#     preprocessing:03cat->purified:__choice__, Type: Categorical, Choices: {encode.one_hot, encode.ordinal, encode.cat_boost}, Default: encode.one_hot
#     preprocessing:03cat->purified:encode.cat_boost:placeholder, Type: Constant, Value: placeholder
#     preprocessing:03cat->purified:encode.one_hot:placeholder, Type: Constant, Value: placeholder
#     preprocessing:03cat->purified:encode.ordinal:placeholder, Type: Constant, Value: placeholder
#     preprocessing:04highR_cat->purified:__choice__, Type: Categorical, Choices: {operate.drop, encode.ordinal, encode.cat_boost}, Default: operate.drop
#     preprocessing:04highR_cat->purified:encode.cat_boost:placeholder, Type: Constant, Value: placeholder
#     preprocessing:04highR_cat->purified:encode.ordinal:placeholder, Type: Constant, Value: placeholder
#     preprocessing:04highR_cat->purified:operate.drop:placeholder, Type: Constant, Value: placeholder
#     preprocessing:05text->tokenized:__choice__, Type: Categorical, Choices: {text.tokenize.simple}, Default: text.tokenize.simple
#     preprocessing:05text->tokenized:text.tokenize.simple:placeholder, Type: Constant, Value: placeholder
#     preprocessing:06tokenized->purified:__choice__, Type: Categorical, Choices: {text.topic.tsvd, text.topic.lsi, text.topic.nmf}, Default: text.topic.tsvd
#     preprocessing:06tokenized->purified:text.topic.lsi:num_topics, Type: UniformInteger, Range: [10, 400], Default: 20, Q: 10
#     preprocessing:06tokenized->purified:text.topic.nmf:num_topics, Type: UniformInteger, Range: [10, 400], Default: 20, Q: 10
#     preprocessing:06tokenized->purified:text.topic.nmf:random_state, Type: Constant, Value: 42:int
#     preprocessing:06tokenized->purified:text.topic.tsvd:num_topics, Type: UniformInteger, Range: [5, 200], Default: 10, Q: 5
#     preprocessing:06tokenized->purified:text.topic.tsvd:random_state, Type: Constant, Value: 42:int
#     preprocessing:07num->scaled:__choice__, Type: Categorical, Choices: {scale.standardize, operate.keep_going}, Default: scale.standardize
#     preprocessing:07num->scaled:operate.keep_going:placeholder, Type: Constant, Value: placeholder
#     preprocessing:07num->scaled:scale.standardize:copy, Type: Constant, Value: False:bool
#     preprocessing:08scaled->purified:__choice__, Type: Categorical, Choices: {operate.keep_going, transform.power}, Default: operate.keep_going
#     preprocessing:08scaled->purified:operate.keep_going:placeholder, Type: Constant, Value: placeholder
#     preprocessing:08scaled->purified:transform.power:placeholder, Type: Constant, Value: placeholder
#     preprocessing:09purified->final:__choice__, Type: Categorical, Choices: {operate.keep_going}, Default: operate.keep_going
#     preprocessing:09purified->final:operate.keep_going:placeholder, Type: Constant, Value: placeholder
#   Conditions:
#     estimating:catboost:border_count | estimating:__choice__ == 'catboost'
#     estimating:catboost:early_stopping_rounds | estimating:__choice__ == 'catboost'
#     estimating:catboost:l2_leaf_reg | estimating:__choice__ == 'catboost'
#     estimating:catboost:learning_rate | estimating:__choice__ == 'catboost'
#     estimating:catboost:max_depth | estimating:__choice__ == 'catboost'
#     estimating:catboost:n_estimators | estimating:__choice__ == 'catboost'
#     estimating:catboost:n_jobs | estimating:__choice__ == 'catboost'
#     estimating:catboost:random_state | estimating:__choice__ == 'catboost'
#     estimating:catboost:subsample | estimating:__choice__ == 'catboost'
#     estimating:catboost:use_best_model | estimating:__choice__ == 'catboost'
#     preprocessing:00highR_nan->nan:operate.drop:placeholder | preprocessing:00highR_nan->nan:__choice__ == 'operate.drop'
#     preprocessing:00highR_nan->nan:operate.keep_going:placeholder | preprocessing:00highR_nan->nan:__choice__ == 'operate.keep_going'
#     preprocessing:01nan->imputed:impute.adaptive_fill:cat_strategy | preprocessing:01nan->imputed:__choice__ == 'impute.adaptive_fill'
#     preprocessing:01nan->imputed:impute.adaptive_fill:num_strategy | preprocessing:01nan->imputed:__choice__ == 'impute.adaptive_fill'
#     preprocessing:02imputed->cat,num:operate.split:column2fg | preprocessing:02imputed->cat,num:__choice__ == 'operate.split'
#     preprocessing:03cat->purified:encode.cat_boost:placeholder | preprocessing:03cat->purified:__choice__ == 'encode.cat_boost'
#     preprocessing:03cat->purified:encode.one_hot:placeholder | preprocessing:03cat->purified:__choice__ == 'encode.one_hot'
#     preprocessing:03cat->purified:encode.ordinal:placeholder | preprocessing:03cat->purified:__choice__ == 'encode.ordinal'
#     preprocessing:04highR_cat->purified:encode.cat_boost:placeholder | preprocessing:04highR_cat->purified:__choice__ == 'encode.cat_boost'
#     preprocessing:04highR_cat->purified:encode.ordinal:placeholder | preprocessing:04highR_cat->purified:__choice__ == 'encode.ordinal'
#     preprocessing:04highR_cat->purified:operate.drop:placeholder | preprocessing:04highR_cat->purified:__choice__ == 'operate.drop'
#     preprocessing:05text->tokenized:text.tokenize.simple:placeholder | preprocessing:05text->tokenized:__choice__ == 'text.tokenize.simple'
#     preprocessing:06tokenized->purified:text.topic.lsi:num_topics | preprocessing:06tokenized->purified:__choice__ == 'text.topic.lsi'
#     preprocessing:06tokenized->purified:text.topic.nmf:num_topics | preprocessing:06tokenized->purified:__choice__ == 'text.topic.nmf'
#     preprocessing:06tokenized->purified:text.topic.nmf:random_state | preprocessing:06tokenized->purified:__choice__ == 'text.topic.nmf'
#     preprocessing:06tokenized->purified:text.topic.tsvd:num_topics | preprocessing:06tokenized->purified:__choice__ == 'text.topic.tsvd'
#     preprocessing:06tokenized->purified:text.topic.tsvd:random_state | preprocessing:06tokenized->purified:__choice__ == 'text.topic.tsvd'
#     preprocessing:07num->scaled:operate.keep_going:placeholder | preprocessing:07num->scaled:__choice__ == 'operate.keep_going'
#     preprocessing:07num->scaled:scale.standardize:copy | preprocessing:07num->scaled:__choice__ == 'scale.standardize'
#     preprocessing:08scaled->purified:operate.keep_going:placeholder | preprocessing:08scaled->purified:__choice__ == 'operate.keep_going'
#     preprocessing:08scaled->purified:transform.power:placeholder | preprocessing:08scaled->purified:__choice__ == 'transform.power'
#     preprocessing:09purified->final:operate.keep_going:placeholder | preprocessing:09purified->final:__choice__ == 'operate.keep_going'

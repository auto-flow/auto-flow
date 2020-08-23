#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import base64
import json
import os

try:
    import Pyro4
except Exception:
    os.system("pip install Pyro4")
try:
    import sympy
except Exception:
    os.system("pip install sympy")

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from autoflow.core.classifier import AutoFlowClassifier
from autoflow.utils.logging_ import setup_logger, get_logger
from autoflow.utils.sys_ import EnvUtils

datapath = os.environ["DATAPATH"]
savedpath = os.getenv("SAVEDPATH", "/home/tqc/Desktop/savedpath")
log_path = f"{savedpath}/autoflow.log"
setup_logger(log_path)
logger = get_logger("main")
envutil = EnvUtils()
envutil.from_json(Path(__file__).parent / "main.json")
envutil.update()
envutil.print(logger)

X_train, y_train, cat = load(datapath)
cat = np.array(cat)
column_descritions = {"cat": X_train.columns[cat].tolist()}
counter = Counter(y_train)
minors = []
for label, count in counter.items():
    if count <= 5:
        minors.append(label)
logger.info(f"minors : {minors}")
mask = ~pd.Series(y_train).isin(minors)
logger.info(f"mask_filtered : {np.count_nonzero(~mask)}")
logger.info(f"column_descritions : {column_descritions}")

y_train = y_train[mask]
X_train = X_train.loc[X_train.index[mask], :]

logger.info(f"nfeatures : {X_train.shape[0]}")
logger.info(f"nrows : {X_train.shape[1]}")
nfeatures = X_train.shape[0]
if 0 <= nfeatures < 1000:
    n_folds = 5
elif 1000 <= nfeatures < 5000:
    n_folds = 3
else:
    n_folds = 1
logger.info(f"n_folds : {n_folds}")

initial_points = [
    {
        "estimating:__choice__": "extra_trees",
        "preprocessing:impute:__choice__": "impute.gbt",
        "preprocessing:impute:missing_rate": 0.8,
        "preprocessing:normed->final:__choice__": "operate.keep_going",
        "preprocessing:num->normed:__choice__": "scale.standard",
        "process_sequence": "impute;num->normed;normed->final",
        "estimating:extra_trees:bootstrap": "False:bool",
        "estimating:extra_trees:criterion": "gini",
        "estimating:extra_trees:early_stopping_rounds": "8:int",
        "estimating:extra_trees:early_stopping_tol": "0.0:float",
        "estimating:extra_trees:iter_inc": "16:int",
        "estimating:extra_trees:max_depth": "None:NoneType",
        "estimating:extra_trees:max_features": "log2",
        "estimating:extra_trees:max_leaf_nodes": "None:NoneType",
        "estimating:extra_trees:min_impurity_decrease": "0:int",
        "estimating:extra_trees:min_samples_leaf": 18,
        "estimating:extra_trees:min_samples_split": 20,
        "estimating:extra_trees:min_weight_fraction_leaf": "0:int",
        "estimating:extra_trees:n_estimators": "1024:int",
        "estimating:extra_trees:n_jobs": "12:int",
        "estimating:extra_trees:random_state": "42:int",
        "preprocessing:impute:impute.gbt:copy": "False:bool",
        "preprocessing:impute:impute.gbt:n_jobs": "12:int",
        "preprocessing:impute:impute.gbt:random_state": "42:int",
        "preprocessing:normed->final:operate.keep_going:placeholder": "placeholder",
        "preprocessing:num->normed:scale.standard:placeholder": "placeholder"
    }
    ]
random_state = envutil.RANDOM_STATE
logger.info(f"random_state : {random_state}")
specific_task_token = envutil.TOKEN
n_workers = envutil.N_WORKERS
# change n_workers
memory_usage = X_train.memory_usage().sum() / 1e6
logger.info(f"X_train memory usage: {memory_usage}M")
if memory_usage > 250:
    logger.info(f" memory_usage > 250, n_workers = 5")
    n_workers = 5
elif memory_usage > 500:
    logger.info(f" memory_usage > 500M, n_workers = 4")
    n_workers = 4
elif memory_usage > 1000:
    logger.info(f" memory_usage > 1G, n_workers = 3")
    n_workers = 3
elif memory_usage > 2000:
    logger.info(f" memory_usage > 2G, n_workers = 2")
    n_workers = 2
elif memory_usage > 5000:
    logger.info(f" memory_usage > 5G, n_workers = 1")
    n_workers = 1
logger.info(f"n_workers = {n_workers}")
if "local_test" in specific_task_token:
    db_params = {
        "user": "tqc",
        "host": "127.0.0.1",
        "port": 5432,
    }
    search_record_db_name = "autoflow_test"
    n_workers = 1
else:
    db_params = {
        "user": "postgres",
        "host": "123.56.90.56",
        "port": 5432,
        "password": "xenon",
    }
    search_record_db_name = "autoflow"

n_jobs_in_algorithm = 12

task_info = envutil.TASK_INFO
if isinstance(task_info, str):
    try:
        task_info = json.loads(base64.b64decode(task_info).decode())
    except Exception as e:
        logger.warning(f"parse task_info failed: {e}")
if not isinstance(task_info, dict):
    task_info = {}
logger.info(f"task_info:\n{json.dumps(task_info, indent=4)}")

kwargs={}
if envutil.ONLY_IMPUTE_GBT:
    kwargs.update(dict(included_imputers="impute.gbt"))

pipe = AutoFlowClassifier(
    store_path=f"/tmp/autoflow",
    imbalance_threshold=2,
    should_record_workflow_step=False,
    save_experiment_model=False,
    del_local_log_path=False,
    db_type="postgresql",
    log_path=log_path,
    db_params=db_params,
    search_record_db_name=search_record_db_name,
    config_generator="ET",
    max_n_samples_for_CV=10000,
    config_generator_params={
        "min_points_in_model": envutil.MIN_POINTS_IN_MODEL,
        "use_local_search": False,
        "use_thompson_sampling": False,
    },
    n_folds=n_folds,
    warm_start=False,
    random_state=random_state,
    min_n_samples_for_SH=50,
    concurrent_type="process",
    n_workers=n_workers,
    SH_only=True,
    min_budget=4,
    max_budget=4,
    n_iterations=envutil.N_ITERATIONS,
    debug_evaluator=False,
    initial_points=None,
    **kwargs
)
pipe.fit(
    X_train, y_train,  # X_test, y_test,
    column_descriptions=column_descritions,
    task_metadata={"openml_task_id": str(envutil.TASK_ID), **task_info},
    specific_task_token=specific_task_token,
    # is_not_realy_run=True,
    fit_ensemble_params=False
)
# score = accuracy_score(y_test, y_pred)
# score = pipe.score(X_test, y_test)
# print(score)

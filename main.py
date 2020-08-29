#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import base64
import json
import os

import psutil

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
savedpath = os.getenv("SAVEDPATH", "/data/savedpath")
log_path = f"{savedpath}/{os.getenv('TASK_ID', 'autoflow')}.log"
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
n_samples = X_train.shape[0]
if 0 <= n_samples < 1000:
    k_folds = 5
elif 1000 <= n_samples < 5000:
    k_folds = 3
else:
    k_folds = 1
logger.info(f"k_folds : {k_folds}")

random_state = envutil.RANDOM_STATE
logger.info(f"random_state : {random_state}")
specific_task_token = envutil.TOKEN
# change n_workers
memory_usage = X_train.memory_usage().sum() / 1e6
logger.info(f"X_train memory usage: {memory_usage}M")
n_workers = envutil.N_WORKERS
if n_workers is None:
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
else:
    db_params = {
        "user": "postgres",
        "host": "123.56.90.56",
        "port": 5432,
        "password": "xenon",
    }
    search_record_db_name = "autoflow"

task_info = envutil.TASK_INFO
if isinstance(task_info, str):
    try:
        task_info = json.loads(base64.b64decode(task_info).decode())
    except Exception as e:
        logger.warning(f"parse task_info failed: {e}")
if not isinstance(task_info, dict):
    task_info = {}
logger.info(f"task_info:\n{json.dumps(task_info, indent=4)}")
vm = psutil.virtual_memory()
total = vm.total / 1024 / 1024
free = vm.free / 1024 / 1024
used = vm.used / 1024 / 1024
task_info.update({
    "n_fold": k_folds,
    "n_workers": n_workers,
    "df_memory_usage": memory_usage,
    "minors_list": minors,
    "total": total,
    "free": free,
    "used": used
})
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
    config_generator_params={
        "min_points_in_model": envutil.MIN_POINTS_IN_MODEL,
        "use_local_search": False,
        "use_thompson_sampling": False,
    },
    evaluation_strategy=None,
    k_folds=k_folds,
    warm_start=False,
    random_state=random_state,
    concurrent_type="process",
    n_workers=n_workers,
    SH_only=True,
    min_budget=1,
    max_budget=1,
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

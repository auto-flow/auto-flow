#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import base64
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from joblib import load
from sklearn.utils.multiclass import type_of_target

from autoflow.core.classifier import AutoFlowClassifier
from autoflow.metrics import roc_auc_macro, log_loss
from autoflow.utils.logging_ import setup_logger, get_logger
from autoflow.utils.sys_ import EnvUtils

datapath = os.environ["DATAPATH"]
savedpath = os.getenv("SAVEDPATH", "/data/savedpath")
mtl_savedpath = f"{savedpath}/ambo"
Path(mtl_savedpath).mkdir(exist_ok=True, parents=True)
log_path = f"{savedpath}/{os.getenv('TASK_ID', 'autoflow')}.log"
setup_logger(log_path)
logger = get_logger("benchmark")
envutil = EnvUtils()
envutil.from_json(Path(__file__).parent / "benchmark.json")
envutil.update()
envutil.print(logger)
X, y, cat, folds = load(datapath)
y = pd.Series(y).infer_objects()
cat = np.array(cat)
fold_id = envutil.FOLD
fold = folds[fold_id]
X_train = X.iloc[fold[0], :]
X_test = X.iloc[fold[1], :]
y_train = y[fold[0]]
y_test = y[fold[1]]

column_descritions = {"cat": X.columns[cat].tolist()}
counter = Counter(y)
minors = []
for label, count in counter.items():
    if count <= 5:
        minors.append(label)
logger.info(f"minors : {minors}")
logger.info(f"column_descritions : {column_descritions}")

logger.info(f"nfeatures : {X.shape[0]}")
logger.info(f"nrows : {X.shape[1]}")

random_state = envutil.RANDOM_STATE
logger.info(f"random_state : {random_state}")
specific_task_token = envutil.TOKEN
# change n_workers
df_memory_usage = X.memory_usage().sum() / 1e6
logger.info(f"X_train memory usage: {df_memory_usage}M")
n_workers = envutil.N_WORKERS
if n_workers is None:
    raise NotImplementedError
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
        "host": os.environ["HOST"],
        "port": 5432,
        "password": os.environ["PASSWORD"],
    }
    search_record_db_name = "autoflow_benchmark"

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
    "n_workers": n_workers,
    "fold_id": fold_id,
    "df_memory_usage": df_memory_usage,
    "minors_list": minors,
    "total": total,
    "free": free,
    "used": used
})

per_run_time_limit = envutil.PER_RUN_TIME_LIMIT
time_left_for_this_task = envutil.TIME_LEFT_FOR_THIS_TASK
warm_start = envutil.WARM_START
# use_workflow_cache = envutil.USE_WORKFLOW_CACHE
if df_memory_usage > 50:
    use_workflow_cache = False
else:
    use_workflow_cache = True
pipe = AutoFlowClassifier(
    store_path=f"{savedpath}/autoflow",  # for resource_manager
    store_dataset=False,  # for resource_manager
    imbalance_threshold=2,
    should_record_workflow_step=False,  # for resource_manager
    save_experiment_model=False,  # for resource_manager
    del_local_log_path=False,  # for resource_manager
    db_type="postgresql",
    log_path=log_path,
    db_params=db_params,  # for resource_manager
    search_record_db_name=search_record_db_name,  # for resource_manager
    config_generator="et-based-ambo",
    config_generator_params={
        "min_points_in_model": envutil.MIN_POINTS_IN_MODEL,
        "use_local_search": envutil.USE_LOCAL_SEARCH,
        "use_thompson_sampling": envutil.USE_THOMPSON_SAMPLING,
        "record_path": mtl_savedpath
    },
    evaluation_strategy="simple",
    refit="dynamic",
    warm_start=warm_start,
    random_state=random_state,
    concurrent_type="process",
    n_workers=n_workers,
    n_iterations=100000,
    debug_evaluator=False,
    initial_points=None,
    use_metalearning=True,
    per_run_time_limit=per_run_time_limit,
    time_left_for_this_task=time_left_for_this_task,
    n_jobs_in_algorithm=1,  # todo: add in env
    use_workflow_cache=use_workflow_cache,
)
type_of_target_ = type_of_target(y)
if type_of_target_ == "binary":
    metric = roc_auc_macro
else:
    metric = log_loss
pipe.fit(
    X_train, y_train, X_test, y_test,
    column_descriptions=column_descritions,
    metric=metric,
    task_metadata={"openml_task_id": str(envutil.TASK_ID), **task_info},
    specific_task_token=specific_task_token,
    fit_ensemble_params=False
)
# score = accuracy_score(y_test, y_pred)
# score = pipe.score(X_test, y_test)
# print(score)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
import random
from pathlib import Path

from ConfigSpace import Configuration
from joblib import load

from autoflow.opt.config_generators.bocg import BayesianOptimizationConfigGenerator
from autoflow.opt.structure import Job
from autoflow.utils.logging_ import setup_logger

setup_logger()
config_space = load("config_space.bz2")
trial_records = json.loads(Path("trial.json").read_text())
bocg = BayesianOptimizationConfigGenerator(config_space, [0, 1 / 16], min_points_in_model=20,
                                           loss_transformer="log_scaled",
                                           config_transformer_params={"impute": -1, "ohe": False})
random.seed(10)
random.shuffle(trial_records)
# warm_start
# budget=0
configs = []
losses = []
for i in range(40):
    record = trial_records[i]
    job = Job(None, **{
        "config": record["config"],
        "config_info": {},
        "budget": 0
    })
    job.result = {"loss": record["loss"]}
    kwargs = {"job": job, "update_model": False}
    if i == 39:
        kwargs["update_model"] = True
    bocg.new_result(**kwargs)
    configs.append(Configuration(config_space, record["config"]))
    losses.append(record["loss"])
for i in range(40, 80):
    record = trial_records[i]
    job = Job(None, **{
        "config": record["config"],
        "config_info": {},
        "budget": 1 / 16
    })
    job.result = {"loss": record["loss"]}
    kwargs = {"job": job, "update_model": False}
    if i == 79:
        kwargs["update_model"] = True
    bocg.new_result(**kwargs)
    configs.append(Configuration(config_space, record["config"]))
    losses.append(record["loss"])
# todo: AF在调用bocg时，热启动后也需要更新权重
bocg.update_weight(force_update=True)
ev_loss = bocg.evaluate(configs, 1 / 16, return_loss=True)
print(ev_loss)

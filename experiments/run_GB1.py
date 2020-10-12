#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from ConfigSpace import Configuration

from autoflow.ambo.config_generators.tree_based import ETBasedAMBO
from autoflow.ambo.structure import Job
from autoflow.feature_engineer.encode import EntityEncoder
from autoflow.hdl.hdl2cs import HDL2CS

# 当前实验配置
experiment_param = dict(
    max_iter=1000,
    repetitions=20,
    random_state=50,
    min_points_in_model=20,
    use_thompson_sampling=2, n_candidates=1000, bandwidth_factor=3, alpha=20, beta=30
)

# 构造超参空间
df = pd.read_csv("GB1.csv")
df2 = df.copy()
df2.index = df2.pop('Variants')
y = df.pop("Fitness")
for i in range(4):
    df[f'X{i}'] = df['Variants'].str[i]
df.pop('Variants')

choices = sorted(Counter(df['X0']).keys())
hdl = {
    f"X{i}": {"_type": "choice", "_value": choices} for i in range(4)
}
config_space = HDL2CS().recursion(hdl)


# 定义目标函数

def evaluation(config: Configuration):
    return -df2.loc["".join([config.get(f"X{i}") for i in range(4)]), 'Fitness']


def main():
    # 定义一个基于文件系统的实验记录方法
    fname = "experiment_params.json"
    if not os.path.exists(fname):
        Path(fname).write_text("[]")
    experiment_params = json.loads(Path(fname).read_text())
    experiment_index = len(experiment_params)
    experiment_params.append(deepcopy(experiment_param))
    Path(fname).write_text(json.dumps(experiment_params, indent=4))
    # 对experiment_param的删除等操作放在存储后面
    base_random_state = experiment_param.pop("random_state")
    repetitions = experiment_param.pop("repetitions")
    min_points_in_model = experiment_param.pop("min_points_in_model")
    max_iter = experiment_param.pop("max_iter")
    res = pd.DataFrame(columns=[f"trial-{i}" for i in range(10)], index=range(max_iter))
    for trial in range(repetitions):
        random_state = base_random_state + trial * 10
        # 设置超参空间的随机种子（会影响后面的采样）
        config_space.seed(random_state)
        # 构造一个具有所有观测的初始样本点集合
        n_repeats = min_points_in_model // 20
        initial_vectors = []
        for i in range(20):
            initial_vectors += [[i] * 4] * n_repeats
        initial_vectors = np.array(initial_vectors)
        rng = np.random.RandomState(random_state)
        for i in range(4):
            rng.shuffle(initial_vectors[:, i])
        initial_points = []
        for vector in initial_vectors:
            initial_points.append(Configuration(config_space, vector=vector))
        encoder_params = dict(max_epoch=100, early_stopping_rounds=50, n_jobs=1, verbose=0)
        print("==========================")
        print(f"= Trial -{trial:01d}-               =")
        print("==========================")
        print('iter |  loss    | config origin')
        print('----------------------------')
        ambo = ETBasedAMBO(
            config_space, [1], random_state=random_state, plot_encoder=False,
            meta_encoder=EntityEncoder(**encoder_params),
            record_path="./", min_points_in_model=min_points_in_model,
            initial_points=initial_points, **experiment_param
        )
        loss = np.inf
        for ix in range(max_iter):
            config, config_info = ambo.get_config(1)
            cur_loss = evaluation(config)
            loss = min(loss, cur_loss)
            print(f" {ix:03d}   {loss:.4f}    {config_info.get('origin')}")
            job = Job("")
            job.result = {"loss": cur_loss}
            job.kwargs = {"budget": 1, "config": config, "config_info": config_info}
            ambo.new_result(job)
            res.loc[ix, f"trial-{trial}"] = cur_loss
    res.to_csv(f"{experiment_index}.csv", index=False)


if __name__ == '__main__':
    main()

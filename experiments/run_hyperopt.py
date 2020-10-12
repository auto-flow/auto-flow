#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
'''
hyperopt  0.1.2
'''
from collections import Counter

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, Configuration
from hyperopt import hp, tpe, fmin, Trials

from autoflow.hdl.hdl2cs import HDL2CS


# 一个将configspace转hyperopt空间的函数
def CS2HyperoptSpace(cs: ConfigurationSpace):
    result = {}
    for hyperparameter in cs.get_hyperparameters():
        name = hyperparameter.name
        if isinstance(hyperparameter, CategoricalHyperparameter):
            result[name] = hp.choice(name, hyperparameter.choices)
        elif isinstance(hyperparameter, UniformFloatHyperparameter):
            lower = hyperparameter.lower
            upper = hyperparameter.upper
            result[name] = hp.uniform(name, lower, upper)
        else:
            raise ValueError
        # todo: 考虑更多情况
    return result


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


# 把ConfigSpace转成hyperopt的空间
space = CS2HyperoptSpace(config_space)
# todo: tpe 参数怎么设置? 是对 tpe.suggest 用偏函数吗？
# todo: hyperopt 没有注释，我对以下参数的理解正确吗？
# 1. n_startup_jobs    冷启动样本数（随机搜索样本数）
# 2. n_EI_candidates   按照l(x)的分布，从样本空间中采样n_EI_candidates个样本，
#                      然后根据EI(x)=l(x)/g(x)计算期望提升值，推荐该值最大的样本
# 3. gamma             默认为0.25，即小于gamma的样本为优势样本，加入到l(x)分布，否则加入g(x)分布
max_iter = 1000
base_random_state = 50
res = pd.DataFrame(columns=[f"trial-{i}" for i in range(10)], index=range(max_iter))

for trial in range(20):
    random_state = base_random_state + trial * 10
    trials = Trials()
    best = fmin(
        evaluation, space, algo=tpe.suggest, max_evals=max_iter,
        rstate=np.random.RandomState(random_state), trials=trials
    )
    losses = trials.losses()
    res[f"trial-{trial}"] = losses
    print(np.min(losses))
res.to_csv("tpe.csv", index=False)

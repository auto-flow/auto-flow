#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
'''
smac  0.12.0
'''

from collections import Counter

# Import ConfigSpace and different types of parameters
import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from autoflow.hdl.hdl2cs import HDL2CS

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


# Scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                     "runcount-limit": 30,  # max. number of function evaluations; for this example set to a low number
                     "cs": config_space,  # configuration space
                     "deterministic": "true"
                     })
smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=evaluation)
incumbent = smac.optimize()
runhistory = smac.runhistory
configs = runhistory.get_all_configs()
losses = [runhistory.get_cost(config) for config in configs]
print(incumbent)

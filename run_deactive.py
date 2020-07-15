#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from joblib import load
from collections import deque
from ConfigSpace import Configuration

config_vector = load("config_vector.bz2")
config_space = load("config_space.bz2")

configuration_space=config_space
configuration=None
vector=config_vector

hyperparameters = configuration_space.get_hyperparameters()
configuration = Configuration(configuration_space=configuration_space,
                              values=configuration,
                              vector=vector,
                              allow_inactive_with_values=True)

hps = deque()

unconditional_hyperparameters = configuration_space.get_all_unconditional_hyperparameters()
hyperparameters_with_children = list()
for uhp in unconditional_hyperparameters:
    children = configuration_space._children_of[uhp]
    if len(children) > 0:
        hyperparameters_with_children.append(uhp)
hps.extendleft(hyperparameters_with_children)

inactive = set()

while len(hps) > 0:
    hp = hps.pop()
    children = configuration_space._children_of[hp]
    for child in children:
        conditions = configuration_space._parent_conditions_of[child.name]
        for condition in conditions:
            if not condition.evaluate_vector(configuration.get_array()):
                dic = configuration.get_dictionary()
                try:
                    del dic[child.name]
                except KeyError:
                    continue
                configuration = Configuration(
                    configuration_space=configuration_space,
                    values=dic,
                    allow_inactive_with_values=True)
                inactive.add(child.name)
            hps.appendleft(child.name)

for hp in hyperparameters:
    if hp.name in inactive:
        dic = configuration.get_dictionary()
        try:
            del dic[hp.name]
        except KeyError:
            continue
        configuration = Configuration(
            configuration_space=configuration_space,
            values=dic,
            allow_inactive_with_values=True)

result= Configuration(configuration_space, values=configuration.get_dictionary())
from copy import deepcopy
import numpy as np
def deactivate(config_space,vector):
    result=deepcopy(vector)
    for i, hp in enumerate(config_space.get_hyperparameters()):
        name=hp.name
        parent_conditions=config_space.get_parent_conditions_of(name)
        parent_condition=parent_conditions[0]
        parent_value=parent_condition.value
        parent_name=parent_condition.parent
        if config_space.get_hyperparameter(parent_name)!=parent_value:
            result[i]=np.nan
    return Configuration(configuration_space=config_space,vector=vector)


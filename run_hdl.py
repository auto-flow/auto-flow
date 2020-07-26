#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.hdl.hdl2shps import HDL2SHPS
from ConfigSpace import CategoricalHyperparameter, Constant
from ConfigSpace import ConfigurationSpace
from ConfigSpace import ForbiddenInClause, ForbiddenEqualsClause, ForbiddenAndConjunction
from ConfigSpace import InCondition, EqualsCondition

hdl={
    "preprocessing":{
        "0num->final(choice)":{
            "scale(choice)":{
                "scale.standardize":{
                    "copy": True
                },
                "scale.normalize":{
                    "copy": True
                },
            },
            "reduce(choice)": {
                "reduce.pca": {
                    "whiten": True
                },
                "reduce.ica": {
                    "whiten": True
                },
                "percent": {"_type": "quniform", "_value": [0, 1, 0.1], "_default": 0.5}
            },
        }
    },
    "estimating(choice)":{
        "lightgbm":{
            "n_estimator":100
        }
    }
}

hdl2shps=HDL2SHPS()
shps=hdl2shps(hdl)
exit(0)
# print(shps)

# Configuration space object:
#   Hyperparameters:
#     estimating:__choice__, Type: Categorical, Choices: {lightgbm}, Default: lightgbm
#     estimating:lightgbm:n_estimator, Type: Constant, Value: 100:int
#     preprocessing:0num->final:__choice__, Type: Categorical, Choices: {scale.standardize}, Default: scale.standardize
#     preprocessing:0num->final:scale.standardize:placeholder, Type: Constant, Value: placeholder
#   Conditions:
#     estimating:lightgbm:n_estimator | estimating:__choice__ == 'lightgbm'
#     preprocessing:0num->final:scale.standardize:placeholder | preprocessing:0num->final:__choice__ == 'scale.standardize'

# scale.standardize
standardize_cs=ConfigurationSpace()
standardize_cs.add_hyperparameter(Constant("copy", "True:bool"))
# scale.normalize
normalize_cs=ConfigurationSpace()
normalize_cs.add_hyperparameter(Constant("copy", "True:bool"))
# scale
scale_cs=ConfigurationSpace()
scale_choice=CategoricalHyperparameter('__choice__', ["scale.standardize", "scale.normalize"])
scale_cs.add_hyperparameter(scale_choice)
scale_cs.add_configuration_space(
    "scale.standardize", standardize_cs, parent_hyperparameter={"parent":scale_choice, "value": "scale.standardize"})
scale_cs.add_configuration_space(
    "scale.normalize", normalize_cs, parent_hyperparameter={"parent":scale_choice, "value": "scale.normalize"})

# reduce.pca
pca_cs=ConfigurationSpace()
pca_cs.add_hyperparameter(Constant("whiten", "True:bool"))
# reduce.ica
ica_cs=ConfigurationSpace()
ica_cs.add_hyperparameter(Constant("whiten", "True:bool"))
# reduce
reduce_cs=ConfigurationSpace()
reduce_choice=CategoricalHyperparameter('__choice__', ["reduce.pca", "reduce.ica"])
reduce_cs.add_hyperparameter(reduce_choice)
reduce_cs.add_configuration_space(
    "reduce.pca", pca_cs, parent_hyperparameter={"parent":reduce_choice, "value": "reduce.pca"})
reduce_cs.add_configuration_space(
    "reduce.ica", ica_cs, parent_hyperparameter={"parent":reduce_choice, "value": "reduce.ica"})

num2target_cs=ConfigurationSpace()
num2target_choice=CategoricalHyperparameter('__choice__', ["scale", "reduce"])
num2target_cs.add_hyperparameter(num2target_choice)
num2target_cs.add_configuration_space("scale", scale_cs, parent_hyperparameter={"parent":num2target_choice, "value": "scale"})
num2target_cs.add_configuration_space("reduce", reduce_cs, parent_hyperparameter={"parent":num2target_choice, "value": "reduce"})

preprocessing_cs=ConfigurationSpace()
preprocessing_cs.add_configuration_space("0num->target", num2target_cs)

cs=ConfigurationSpace()
cs.add_configuration_space("preprocessing", preprocessing_cs)
print(cs)























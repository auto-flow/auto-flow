#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

from hyperflow.hdl.hdl_constructor import HDL_Constructor
from hyperflow.manager.data_manager import DataManager
import numpy as np

data_manager = DataManager(X_train=np.random.rand(3,3), y_train=np.arange(3))
hdl_constructor = HDL_Constructor(DAG_workflow={"num->target":["lightgbm"]},
           hdl_bank={"classification":{"lightgbm":{"boosting_type":  {"_type": "choice", "_value":["gbdt","dart","goss"]}}}})

hdl_constructor.run(data_manager,42,0.5)
print(hdl_constructor.hdl)


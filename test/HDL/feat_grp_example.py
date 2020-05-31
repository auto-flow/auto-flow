#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import os

import numpy as np
import pandas as pd

from autoflow.hdl.hdl_constructor import HDL_Constructor
from autoflow.manager.data_manager import DataManager
from autoflow.manager.data_container.dataframe import DataFrameContainer

X = pd.DataFrame(np.random.rand(3, 3),columns=['c1','c2','c3'])
column_descriptions={
    "2d":"c1",
    "3d":"c2",
    "AD2D":"c3"
}

data_manager = DataManager(X_train=X, y_train=np.arange(3),column_descriptions=column_descriptions)
hdl_constructor = HDL_Constructor(
    DAG_workflow={
        "2d->final": ["operate.drop", "operate.keep_going"],
        "3d->final": ["operate.drop", "operate.keep_going"],
        "AD2D->final": ["operate.drop", "operate.keep_going"],
        "final->target": ["lightgbm", "libsvm_svc", "random_forest"]
    }
)

hdl_constructor.run(data_manager)
graph = hdl_constructor.draw_workflow_space()
open("workflow_space.gv", "w+").write(graph.source)
cmd = f'''dot -Tpng -Gsize=9,15\! -Gdpi=300 -oworkflow_space.png workflow_space.gv'''
os.system(cmd)

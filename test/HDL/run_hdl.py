#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import os

import numpy as np
import pandas as pd

from autoflow.hdl.hdl_constructor import HDL_Constructor
from autoflow.manager.data_manager import DataManager

X = pd.DataFrame(np.random.rand(3, 3))

X.iloc[0, 0] = "a"
data_manager = DataManager(X_train=X, y_train=np.arange(3))
hdl_constructor = HDL_Constructor(
    DAG_workflow={
        "num->scaled": ["scale.minmax", "scale.standardize", None],
        "scaled->transformed": ["scale.normalize", "transform.power", None],
        "transformed->selected": ["select.ref_clf", "select.from_model_clf", None],
        "selected->final": ["reduce.pca", None],
        "cat->final": ["encode.ordinal", "encode.target"],
        "final->target": ["lightgbm", "libsvm_svc", "random_forest"]
    }
)

hdl_constructor.run(data_manager, 42, 0.5)
graph = hdl_constructor.draw_workflow_space()
open("workflow_space.gv", "w+").write(graph.source)
cmd = f'''dot -Tpng -Gsize=9,15\! -Gdpi=300 -oworkflow_space.png workflow_space.gv'''
os.system(cmd)

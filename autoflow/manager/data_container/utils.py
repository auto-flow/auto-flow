#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from typing import Optional

from autoflow.manager.data_container.base import DataContainer


def copy_data_container_structure(obj: Optional[DataContainer]):
    if obj is None:
        return None
    data = obj.data
    obj.data = None
    obj_ = deepcopy(obj)
    obj.data = data
    return obj_


if __name__ == '__main__':
    from autoflow.manager.data_container.dataframe import DataFrameContainer
    import pandas as pd
    import numpy as np

    data = np.array([[1, 32, 3], [4, 5, 7]])
    X = DataFrameContainer("TestSet", dataset_instance=data)
    X2=copy_data_container_structure(X)
    X2.data=X.data
    X2.set_feature_groups(["a","a","b"])
    print(X)
    print(X2)

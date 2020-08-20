#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from collections import namedtuple

from sklearn.utils.multiclass import type_of_target


class MLTask(namedtuple("Task", ["mainTask", "subTask", "role"])):
    pass


def get_ml_task_from_y(y):
    from autoflow.constants import binary_classification_task, multiclass_classification_task, \
        multilabel_classification_task, regression_task
    y_type = type_of_target(y)
    if y_type == "binary":
        ml_task = binary_classification_task
    elif y_type == "multiclass":
        ml_task = multiclass_classification_task
    elif y_type == "multilabel-indicator":
        ml_task = multilabel_classification_task
    elif y_type == "multiclass-multioutput":
        raise NotImplementedError()
    elif y_type == "continuous":
        ml_task = regression_task
    else:
        raise NotImplementedError()
    return ml_task

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import inspect
from typing import Dict, Any, Sequence


def get_valid_params_in_kwargs(klass, kwargs: Dict[str, Any]):
    validated = {}
    for key, value in kwargs.items():
        if key in inspect.signature(klass.__init__).parameters.keys():
            validated[key] = value
    return validated


def instancing(variable, klass, kwargs):
    if variable is None:
        variable = klass(**get_valid_params_in_kwargs(klass, kwargs))
    elif isinstance(variable, dict):
        variable = klass(**variable)
    elif isinstance(variable, klass):
        pass
    elif isinstance(variable, Sequence):
        for elem in variable:
            assert isinstance(elem, klass)
    else:
        raise NotImplementedError
    return variable


def sequencing(variable, klass):
    if not isinstance(variable, Sequence):
        variable = [variable]
    variables: Sequence[klass] = variable
    return variables
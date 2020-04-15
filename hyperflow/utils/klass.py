#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import inspect
from typing import Dict, Any, Sequence


class StrSignatureMixin():
    def __str__(self):
        result = f"{self.__class__.__name__}("
        valid_params = []
        for key in inspect.signature(self.__init__).parameters.keys():
            value = getattr(self, key, "NaN")
            if value != "NaN":
                valid_params.append([key, value])
        valid_params_str_list = [f"{key}={repr(value)}" for key, value in valid_params]
        N = len(valid_params_str_list)
        if N > 0:
            result += "\n"
        for i, valid_params_str in enumerate(valid_params_str_list):
            result += f"\t{valid_params_str}"
            if i == N - 1:
                result += ", "
            result += "\n"
        result += ")"
        return result

    def __repr__(self):
        return self.__str__()


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

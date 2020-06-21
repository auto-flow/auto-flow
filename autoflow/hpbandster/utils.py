#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

from typing import Dict

import numpy as np


def get_max_SH_iter(min_budget, max_budget, eta):
    return -int(np.log(min_budget / max_budget) / np.log(eta)) + 1


def modify_timestamps(timestamps: Dict[str, float], delta: float) -> Dict[str, float]:
    result = {}
    for k, v in timestamps.items():
        v += delta
        result[k] = v
    return result

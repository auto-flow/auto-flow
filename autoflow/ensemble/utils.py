#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from typing import List

import numpy as np


def vote_predicts(predicts: List[np.ndarray]):
    probas_arr = np.array(predicts)
    proba = np.average(probas_arr, axis=0)
    return proba


def mean_predicts(predicts: List[np.ndarray]):
    probas_arr = np.array(predicts)
    proba = np.average(probas_arr, axis=0)
    return proba
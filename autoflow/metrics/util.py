#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.metrics import Scorer


def score2loss(score: float, metric: Scorer) -> float:
    return metric._optimum - metric._sign * score

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from fractions import Fraction
from typing import Dict

import numpy as np

from autoflow.utils.logging_ import get_logger

inc_logger = get_logger("incumbent trajectory")


def get_max_SH_iter(min_budget, max_budget, eta):
    return -int(np.log(min_budget / max_budget) / np.log(eta)) + 1


def modify_timestamps(timestamps: Dict[str, float], delta: float) -> Dict[str, float]:
    result = {}
    for k, v in timestamps.items():
        v += delta
        result[k] = v
    return result


def print_incumbent_trajectory(chal_perf: float, inc_perf: float, challenger: dict, incumbent: dict, budget: float):
    inc_logger.info("Challenger (%.4f) is better than incumbent (%.4f) when budget is (%s)."
                    % (chal_perf, inc_perf, pprint_budget(budget)))
    # Show changes in the configuration
    params = sorted([(param, incumbent.get(param), challenger.get(param))
                     for param in challenger.keys()])
    inc_logger.info("Changes in incumbent:")
    for param in params:
        if param[1] != param[2]:
            inc_logger.info("  %s : %r -> %r" % (param))
        else:
            inc_logger.debug("  %s remains unchanged: %r" %
                             (param[0], param[1]))


def pprint_budget(budget: float):
    if budget - float(int(budget)) == 0:
        return str(int(budget))
    fraction = Fraction.from_float(budget)
    return f"{fraction.numerator}/{fraction.denominator}"

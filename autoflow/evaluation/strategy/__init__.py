#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import re

import numpy as np

from autoflow.utils.logging_ import get_logger
from autoflow.utils.ml_task import MLTask

logger = get_logger(__package__)


def parse_evaluation_strategy(evaluation_strategy, SH_holdout_condition, n_samples, n_features, n_classes,
                              ml_task: MLTask) -> dict:
    '''

    Parameters
    ----------
    evaluation_strategy
    SH_holdout_condition
    n_samples
    n_features
    n_classes
    ml_task

    Returns
    -------
    k_folds, eta, min_budget, max_budget
    '''
    # todo: n_iteration, HB
    strategies = [
        'SH-5CV',
        '5CV',
        '3CV',
        'SH-holdout',
        'holdout',
        '10CV',
        'SH-3CV',
        'SH-10CV']
    s = n_samples
    f = n_features
    is_SH_holdout = False
    if bool(SH_holdout_condition):
        try:
            is_SH_holdout = eval(SH_holdout_condition)
            logger.info(f"SH_holdout_condition | ({SH_holdout_condition}) = {is_SH_holdout}")
        except Exception as e:
            logger.warning(f"eval(SH_holdout_condition) failed, SH_holdout_condition = '{SH_holdout_condition}'")
            logger.warning(str(e))
    if is_SH_holdout and evaluation_strategy in ("simple", "auto"):
        logger.info("`SH_holdout_condition` is satisfied, using 'SH-holdout' evaluation strategy.")
        return {
            "k_folds": 1,
            "eta": 4,
            "min_budget": 1 / 16,
            "max_budget": 1,
            "SH_only": True
        }
    if evaluation_strategy == "auto":
        if ml_task.mainTask == "regression":
            # todo implement regression-auto
            logger.info(f"didn't support auto predict regression problem's evaluation_strategy, "
                        f"using 'simple' instead.")
        else:
            from .auto import askl2  # when import , strategy selector is auto fit.
            predictions = askl2.selector.predict(np.array([n_classes, n_features, n_samples]),
                                                 soft=True)
            evaluation_strategy = strategies[(predictions).flatten().argmax()]
            logger.info(f"evaluation_strategy is predicted to '{evaluation_strategy}' automatically")
    if evaluation_strategy == "simple":
        return {
            "k_folds": 3,
            "eta": 4,
            "min_budget": 1 / 16,
            "max_budget": 1,
            "SH_only": True
        }
    # 解析 evaluate_strategy
    value_error = ValueError(f"Unrecognizable evaluation_strategy {evaluation_strategy}")
    assert isinstance(evaluation_strategy, str), value_error
    pattern = re.compile("^(?P<SH>SH\-)?(?P<CV_holdout>((?P<k_folds>\d+)CV)|holdout)$")
    m = pattern.match(evaluation_strategy)
    assert m is not None, value_error
    CV_holdout = m.group("CV_holdout")
    if CV_holdout == "holdout":
        k_folds = 1
    else:
        k_folds = m.group("k_folds")
        assert k_folds is not None
        k_folds = int(k_folds)
        assert k_folds >= 1
    SH = m.group("SH")
    if SH is None:
        min_budget = 1
        max_budget = 1
    else:
        min_budget = 1 / 16
        max_budget = 1
    return {
        "k_folds": k_folds,
        "eta": 4,
        "min_budget": min_budget,
        "max_budget": max_budget,
        "SH_only": True
    }

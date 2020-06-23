#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import json5 as json
import numpy as np
from joblib import Memory

from autoflow.constants import JOBLIB_CACHE, ITERATIONS_BUDGET_MODE, SUBSAMPLES_BUDGET_MODE
from autoflow.data_container import DataFrameContainer, NdArrayContainer
from autoflow.utils.array_ import get_stratified_sampling_index
from autoflow.utils.packages import find_components
from autoflow.workflow.components.base import AutoFlowIterComponent
from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm


def implement_subsample_budget(
        X_train: DataFrameContainer, y_train: NdArrayContainer,
        Xs: List[Optional[DataFrameContainer]],
        budget, random_state: int
) -> Tuple[DataFrameContainer, NdArrayContainer, List[Optional[DataFrameContainer]]]:
    rng = np.random.RandomState(random_state)
    samples = round(X_train.shape[0] * budget)
    features = X_train.shape[1]
    sub_sample_index = get_stratified_sampling_index(y_train.data, budget, random_state)
    # sub sampling X_train, y_train
    X_train = X_train.sub_sample(sub_sample_index)
    y_train = y_train.sub_sample(sub_sample_index)
    # if features > samples , do sub_feature avoid over-fitting
    if features > samples:
        sub_feature_index = rng.permutation(X_train.shape[1])[:samples]
        X_train = X_train.sub_feature(sub_feature_index)
        res_Xs = []
        for X in Xs:
            res_Xs.append(X.sub_feature(sub_feature_index) if X is not None else None)
    else:
        res_Xs = Xs
    return X_train, y_train, res_Xs


def _get_default_algo2budget_mode() -> Dict[str, str]:
    import autoflow.workflow.components.classification as clf
    import autoflow.workflow.components.regression as reg
    algo2budget_mode = {}
    for module, base_cls in zip(
            (clf, reg),
            (AutoFlowClassificationAlgorithm, AutoFlowRegressionAlgorithm)
    ):
        directory = os.path.split(module.__file__)[0]
        result = find_components(
            module.__package__,
            directory,
            base_cls
        )
        for cls in result.values():
            class_name = cls.__name__
            if issubclass(cls, AutoFlowIterComponent) or getattr(cls, "support_early_stopping", True):
                budget_mode = ITERATIONS_BUDGET_MODE
            else:
                budget_mode = SUBSAMPLES_BUDGET_MODE
            algo2budget_mode[class_name] = budget_mode
    return algo2budget_mode


memory = Memory(JOBLIB_CACHE, verbose=1)
get_default_algo2budget_mode = memory.cache(_get_default_algo2budget_mode)


def _get_default_algo2iter():
    return json.loads((Path(__file__).parent / "algo2iter.json").read_text())


get_default_algo2iter = memory.cache(_get_default_algo2iter)

if __name__ == '__main__':
    from pprint import pprint

    pprint(_get_default_algo2iter())

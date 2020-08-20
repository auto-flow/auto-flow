#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
# todo 早停策略MixIn
from autoflow.workflow.components.iter_algo import TabularNNIterativeMixIn
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm
import numpy as np

__all__ = ["TabularNNRegressor"]


class TabularNNRegressor(TabularNNIterativeMixIn, AutoFlowRegressionAlgorithm):
    class__ = "TabularNNRegressor"
    module__ = "autoflow.estimator.tabular_nn_est"

    support_early_stopping = True



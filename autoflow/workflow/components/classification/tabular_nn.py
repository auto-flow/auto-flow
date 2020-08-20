#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
# todo 早停策略MixIn
import numpy as np

from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm
from autoflow.workflow.components.iter_algo import TabularNNIterativeMixIn

__all__ = ["TabularNNClassifier"]


class TabularNNClassifier(TabularNNIterativeMixIn, AutoFlowClassificationAlgorithm):
    class__ = "TabularNNClassifier"
    module__ = "autoflow.estimator.tabular_nn_est"

    support_early_stopping = True



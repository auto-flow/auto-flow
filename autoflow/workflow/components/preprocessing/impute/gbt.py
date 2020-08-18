#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from .base import BaseImputer

__all__ = ["GBTImputer"]


class GBTImputer(BaseImputer):
    class__ = "GBTImputer"
    module__ = "autoflow.feature_engineer.impute"
    need_y = True
    additional_info_keys = ("iter", "gamma_history", "gamma_cat_history", "cost_times")

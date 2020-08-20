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

    @property
    def additional_info(self):
        return dict(
            self.form_additional_info_pair(key) for key in
            list(self.additional_info_keys) + ["iter", "gamma_history", "gamma_cat_history", "cost_times"]
        )

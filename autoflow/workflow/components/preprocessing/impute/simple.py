#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from .base import BaseImputer

__all__ = ["SimpleImputer"]


class SimpleImputer(BaseImputer):
    class__ = "SimpleImputer"
    module__ = "autoflow.feature_engineer.impute"

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.workflow.components.preprocessing.encode.base import BaseEncoder

__all__ = ["EntityEncoder"]


class EntityEncoder(BaseEncoder):
    class__ = "EntityEncoder"
    module__ = "autoflow.feature_engineer.encode"
    need_y = True
    cache_intermediate = True
    additional_info_keys = ("iter",)

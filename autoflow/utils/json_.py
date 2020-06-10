#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import datetime
import json


# from collections import defaultdict


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return str(obj)
        elif isinstance(obj, bytes):
            # todo: base64
            return obj.decode(encoding="utf-8")
        # elif isinstance(obj,defaultdict):

        else:
            return json.JSONEncoder.default(self, obj)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from generic_fs.hdfs_ import HDFS
from generic_fs.local import LocalFS
from generic_fs.nitrogen import NitrogenFS


def get_file_system(file_system_name):
    if file_system_name == "local":
        return LocalFS
    elif file_system_name == "hdfs":
        return HDFS
    elif file_system_name == "s3":
        raise NotImplementedError
    elif file_system_name=="nitrogen":
        return NitrogenFS
    else:
        raise NotImplementedError

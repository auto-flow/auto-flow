#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import pickle
from typing import Any



class BaseCache():
    def __init__(self, resource_manager):
        from autoflow.resource_manager.base import ResourceManager
        self.res:ResourceManager = resource_manager

    def get(self, k: str) -> Any:
        raise NotImplementedError

    def set(self, k: str, v: Any):
        raise NotImplementedError


class FsCache(BaseCache):
    def __init__(self, resource_manager):
        super(FsCache, self).__init__(resource_manager)
        # todo: 增加一个参数支持用户自定义文件系统
        self.cache_dir = self.res.file_system.join(
            self.res.store_path,
            "cache"
        )
        self.res.file_system.mkdir(self.cache_dir)

    def k2path(self, k):
        return self.res.file_system.join(
            self.cache_dir,
            ".".join([k, self.res.compress_suffix])
        )

    def get(self, k: str) -> Any:
        path = self.k2path(k)
        if self.res.file_system.exists(path):
            return self.res.file_system.load_pickle(path)
        return None

    def set(self, k: str, v: Any):
        path = self.k2path(k)
        self.res.file_system.dump_pickle(v, path)


class RedisCache(BaseCache):

    def get(self, k: str) -> Any:
        v = self.res.redis_get(k)
        if v is None:
            return None
        return pickle.loads(v)

    def set(self, k: str, v: Any):
        self.res.redis_set(k, pickle.dumps(v))

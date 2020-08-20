#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import logging
import pickle
from multiprocessing import Manager
from time import sleep
from typing import Any

lock = Manager().dict()
logger = logging.getLogger("CacheLock")


class BaseCache():
    def __init__(self, resource_manager):
        from autoflow.resource_manager.base import ResourceManager
        self.res: ResourceManager = resource_manager

    def get(self, k: str) -> Any:
        pass
        # while True:
        #     if lock.get(k) == "w":
        #         logger.warning(f"key[{k}] is writing, sleep 0.1s ...")
        #         sleep(0.1)
        #     else:
        #         break

    def set(self, k: str, v: Any):
        pass
        # lock[k] = "w"

    def release_k(self, k):
        pass
        # lock.pop(k, None)


class FsCache(BaseCache):
    def __init__(self, resource_manager):
        super(FsCache, self).__init__(resource_manager)
        # todo: 增加一个参数支持用户自定义文件系统
        self.cache_dir = self.res.file_system.join(
            self.res.store_path,
            "cache"
        )
        self.res.file_system.mkdir(self.cache_dir)
        self.max_trials = 3

    def k2path(self, k, suffix=None):
        if suffix is None:
            suffix = self.res.compress_suffix
        return self.res.file_system.join(
            self.cache_dir,
            ".".join([k, suffix])
        )

    def wait_lock(self, lock_path):
        tol = 0
        while True:
            if self.res.file_system.exists(lock_path):
                if tol >= 10:
                    logger.error(f"lock[{lock_path}] wait 10 times, break.")
                    break
                logger.warning(f"lock[{lock_path}] is writing, sleep 0.1s ...")
                sleep(0.1)
                tol += 1
            else:
                break

    def get(self, k: str) -> Any:
        super(FsCache, self).get(k)
        path = self.k2path(k)
        lock_path = self.k2path(k, "lock")
        if self.res.file_system.exists(path):
            self.wait_lock(lock_path) # wait lock
            value = None
            for trial in range(self.max_trials):
                try:
                    value = self.res.file_system.load_pickle(path)
                    break
                except Exception as e:
                    logger.error(e)
            return value
        return None

    def set(self, k: str, v: Any):
        super(FsCache, self).set(k, v)
        lock_path = self.k2path(k, "lock")
        self.wait_lock(lock_path) # wait lock
        self.res.file_system.touch_file(lock_path) # add lock
        path = self.k2path(k)
        self.res.file_system.dump_pickle(v, path)
        # self.release_k(k)
        try:
            self.res.file_system.delete(lock_path) # release lock
        except Exception as e:
            logger.warning(f"exception when release lock: {e}")



class RedisCache(BaseCache):

    def get(self, k: str) -> Any:
        super(RedisCache, self).get(k)
        v = self.res.redis_get(k)
        if v is None:
            return None
        return pickle.loads(v)

    def set(self, k: str, v: Any):
        super(RedisCache, self).set(k, v)
        self.res.redis_set(k, pickle.dumps(v))
        self.release_k(k)

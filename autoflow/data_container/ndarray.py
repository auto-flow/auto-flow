#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import hashlib
from copy import deepcopy

from autoflow.data_container.base import DataContainer
import numpy as np

from autoflow.utils.hash import get_hash_of_str, get_hash_of_array


class NdArrayContainer(DataContainer):
    VALID_INSTANCE = np.ndarray
    dataset_type = "ndarray"


    def process_dataset_instance(self, dataset_instance):
        return dataset_instance

    def get_hash(self):
        m = hashlib.md5()
        get_hash_of_str(self.dataset_type, m)
        get_hash_of_str(self.dataset_source, m)
        return get_hash_of_array(self.data, m)  # todo ： 测试在存储与加载后hash值是否会改变

    def upload(self, upload_type="fs"):
        self.dataset_hash = self.get_hash()
        if self.dataset_hash==self.uploaded_hash:
            return
        L, dataset_id, dataset_path = self.resource_manager.insert_to_dataset_table(
            self.dataset_hash, self.dataset_metadata, "fs", self.dataset_source, {},
            {}, [])
        if L != 0:
            self.logger.info(f"Dataset ID: {dataset_id} is already exists, {self.dataset_source} will not upload. ")
        else:
            self.resource_manager.upload_ndarray_to_fs(self.data, dataset_path)
        super(NdArrayContainer, self).upload(upload_type)

    def download(self, dataset_id):
        records = self.resource_manager.query_dataset_record(dataset_id)
        if len(records) == 0:
            raise ValueError(f"dataset_id: {dataset_id} didn't exists.")
        record = records[0]
        self.dataset_source = record["dataset_source"]
        self.dataset_metadata = record["dataset_metadata"]
        arr = self.resource_manager.download_arr_of_fs(dataset_id)
        self.arr=arr
    def read_local(self, path: str):
        # todo: 加载 npy 或 h5
        pass

    def sub_sample(self, index):
        new_arr = self.copy()
        new_arr.data = deepcopy(self.data[index])  # fixme: 是否有copy的必要？
        return new_arr
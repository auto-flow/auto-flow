#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import hashlib
from copy import deepcopy

import numpy as np
import pandas as pd

from autoflow.data_container.base import DataContainer
from autoflow.utils.hash import get_hash_of_str, get_hash_of_array


class NdArrayContainer(DataContainer):
    VALID_INSTANCE = (np.ndarray, list, tuple, pd.Series)
    dataset_type = "ndarray"

    def process_dataset_instance(self, dataset_instance):
        if isinstance(dataset_instance, np.ndarray):
            return dataset_instance
        else:
            return np.array(dataset_instance)

    def get_hash(self):
        m = hashlib.md5()
        get_hash_of_str(self.dataset_type, m)
        get_hash_of_str(self.dataset_source, m)
        return get_hash_of_array(self.data, m)

    def upload(self, upload_type="fs"):
        self.dataset_hash = self.get_hash()
        if self.dataset_hash == self.uploaded_hash:
            return
        respond = self.resource_manager.insert_to_dataset_table(
            self.dataset_hash, self.dataset_metadata, "fs", self.dataset_source, {},
            {}, [])
        L, dataset_id, dataset_path = respond["length"], respond["dataset_id"], respond["dataset_path"]
        if L != 0:
            self.logger.info(f"Dataset ID: {dataset_id} is already exists, {self.dataset_source} will not upload. ")
        else:
            self.resource_manager.upload_ndarray_to_fs(self.data, dataset_path)
        super(NdArrayContainer, self).upload(upload_type)

    def download(self, dataset_id):
        records = self.resource_manager.get_dataset_records(dataset_id)
        if len(records) == 0:
            raise ValueError(f"dataset_id: {dataset_id} didn't exists.")
        record = records[0]
        self.dataset_source = record["dataset_source"]
        self.dataset_metadata = record["dataset_metadata"]
        arr = self.resource_manager.download_arr_from_fs(record["dataset_path"])
        self.data = arr

    def read_local(self, path: str):
        # todo: 加载 npy 或 h5
        pass

    def sub_sample(self, index):
        new_arr = self.copy()
        new_arr.data = deepcopy(self.data[index])  # fixme: 是否有copy的必要？
        return new_arr

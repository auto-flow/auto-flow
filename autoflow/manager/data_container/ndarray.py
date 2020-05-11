#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.manager.data_container.base import DataContainer
import numpy as np

class NdArrayContainer(DataContainer):
    VALID_INSTANCE = np.ndarray
    dataset_type = "ndarray"


    def process_dataset_instance(self, dataset_instance):
        return dataset_instance

    def get_hash(self):
        pass

    def upload(self):
        pass

    def download(self, dataset_id):
        pass

    def read_local(self, path: str):
        pass

    def sub_sample(self, index):
        return self.data[index]
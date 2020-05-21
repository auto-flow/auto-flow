#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from pickle import dumps
from typing import Optional

import numpy as np
from frozendict import frozendict

from autoflow.utils.logging_ import get_logger


class DataContainer():
    VALID_INSTANCE = None
    dataset_type = None

    def __init__(self, dataset_source="", dataset_path=None, dataset_instance=None, dataset_id=None,
                 resource_manager=None,
                 dataset_metadata=frozendict()):
        self.dataset_hash = None
        self.dataset_source = dataset_source
        self.dataset_metadata = dict(dataset_metadata)
        self.dataset_metadata.update(dataset_source=dataset_source)
        self.uploaded_hash = None
        from autoflow.manager.resource_manager import ResourceManager
        self.logger = get_logger(self)
        if resource_manager is None:
            self.logger.warning(
                "In DataContainer __init__, resource_manager is None, create a default local resource_manager.")
            resource_manager = ResourceManager()
        self.resource_manager: ResourceManager = resource_manager
        data_indicators = [dataset_path, dataset_instance, dataset_id]
        data_indicators = np.array(list(map(lambda x: x is not None, data_indicators)), dtype='int32')
        assert data_indicators.sum() == 1
        if dataset_path is not None:
            data = self.read_local(dataset_path)
            self.data = self.process_dataset_instance(data)
        elif dataset_instance is not None:
            assert isinstance(dataset_instance, self.VALID_INSTANCE)
            self.data = self.process_dataset_instance(dataset_instance)
        elif dataset_id is not None:
            self.data = self.download(dataset_id)
        else:
            raise NotImplementedError

    def process_dataset_instance(self, dataset_instance):
        raise NotImplementedError

    def get_hash(self):
        raise NotImplementedError

    def upload(self, upload_type):
        self.uploaded_hash = self.get_hash()

    def download(self, dataset_id):
        raise NotImplementedError

    def read_local(self, path: str):
        raise NotImplementedError

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    def __str__(self):
        return f"{self.__class__.__name__}: \n" + str(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__}: \n" + repr(self.data)

    def copy(self):
        if self is None:
            return None
        data = self.data
        rm = self.resource_manager
        self.data = None
        self.resource_manager = None
        self_res = deepcopy(self)
        self_res.resource_manager = rm
        self_res.data = data
        self.data=data
        self.resource_manager=rm
        return self_res

    def pickle(self):
        if self is None:
            return None
        data = self.data
        rm = self.resource_manager
        self.data = None
        self.resource_manager = None
        self_res = dumps(self)
        self_res.resource_manager = rm
        self_res.data = data
        return self_res

    def sub_sample(self, index):
        raise NotImplementedError




def get_container_data(X: DataContainer):
    if X is None:
        return None
    return X.data

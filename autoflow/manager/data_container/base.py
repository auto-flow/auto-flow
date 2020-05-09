#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
from frozendict import frozendict

from autoflow.utils.logging_ import get_logger


class DataContainer():
    VALID_INSTANCE = None

    def __init__(self, dataset_type, dataset_path=None, dataset_instance=None, dataset_id=None, resource_manager=None,
                 dataset_metadata=frozendict()):
        self.dataset_type = dataset_type
        self.dataset_metadata = dict(dataset_metadata)
        self.dataset_metadata.update(dataset_type=dataset_type)
        from autoflow.manager.resource_manager import ResourceManager
        self.logger = get_logger(self)
        if resource_manager is None:
            self.logger.warning(
                "In DataContainer __init__, resource_manager is None, create a default local resource_manager.")
            resource_manager = ResourceManager()
        self.resource_manager = resource_manager
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

    def upload(self):
        raise NotImplementedError

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

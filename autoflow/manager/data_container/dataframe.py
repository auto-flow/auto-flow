#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import hashlib
from copy import deepcopy

import inflection
import numpy as np
import pandas as pd
from frozendict import frozendict

from autoflow.constants import VARIABLE_PATTERN
from autoflow.manager.data_container.base import DataContainer
from autoflow.utils.dataframe import inverse_dict
from autoflow.utils.hash import get_hash_of_dataframe, get_hash_of_dict


class DataframeDataContainer(DataContainer):
    VALID_INSTANCE = (np.ndarray, pd.DataFrame)

    def __init__(self, dataset_path=None, dataset_instance=None, dataset_id=None, resource_manager=None,
                 dataset_metadata=frozendict()):
        self.column_descriptions = None
        super(DataframeDataContainer, self).__init__(dataset_path, dataset_instance, dataset_id, resource_manager,
                                                     dataset_metadata)

    def process_dataset_instance(self, dataset_instance):
        if isinstance(dataset_instance, np.ndarray):
            dataset_instance = pd.DataFrame(dataset_instance,
                                            columns=[f"column_{i}" for i in range(len(dataset_instance.shape[1]))])
        elif isinstance(dataset_instance, pd.DataFrame):
            origin_columns = dataset_instance.columns
            columns = deepcopy(origin_columns)
            for column in columns:
                if not VARIABLE_PATTERN.match(column):
                    raise ValueError(f"Column '{column}' in DataFrame is invalid.")
            columns = [inflection.underscore(column).lower() for column in columns]

            unique_col, counts = np.unique(columns, return_counts=True)
            dup_col = unique_col[counts > 1]
            if len(dup_col) > 0:
                raise ValueError(f"Column {dup_col} are duplicated!")
            self.columns_mapper = dict(zip(origin_columns, columns))
        else:
            raise NotImplementedError
        return dataset_instance

    def get_hash(self):
        assert self.column_descriptions is not None
        m = hashlib.md5()
        get_hash_of_dict(self.column_descriptions, m)
        return get_hash_of_dataframe(self.data, m)

    def read_local(self, path: str):
        if path.endswith(".csv"):
            return pd.read_csv(path)
        else:
            raise NotImplementedError

    def upload(self):
        self.dataset_hash = self.get_hash()
        L, dataset_id = self.resource_manager.insert_to_dataset_table(
            self.dataset_hash, self.dataset_metadata, "dataframe", self.column_descriptions, self.columns_mapper)
        if L != 0:
            self.logger.info(f"Dataset ID: {dataset_id} is already exists, will not upload. ")
        else:
            self.resource_manager.upload_df_to_table(self.data, self.dataset_hash)

    def download(self, dataset_id):
        records = self.resource_manager.query_dataset_record(dataset_id)
        if len(records) == 0:
            raise ValueError(f"dataset_id: {dataset_id} didn't exists.")
        record = records[0]
        self.column_descriptions = record["column_descriptions"]
        self.columns_mapper = record["columns_mapper"]
        df = self.resource_manager.download_df_of_table(dataset_id)
        inverse_columns_mapper = inverse_dict(self.columns_mapper)
        df.columns = df.columns.map(inverse_columns_mapper)
        return df

    @property
    def columns(self):
        return self.data.columns

    @columns.setter
    def columns(self, columns_):
        self.data.columns = columns_

    @property
    def index(self):
        return self.data.index

    @index.setter
    def index(self, index_):
        self.data.index = index_

    @property
    def dtypes(self):
        return self.data.dtypes

    @dtypes.setter
    def dtypes(self, dtypes_):
        self.data.dtypes = dtypes_

    def __str__(self):
        return f"{self.__class__.__name__}: \n" + str(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__}: \n" + repr(self.data)

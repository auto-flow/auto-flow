#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import hashlib
from copy import deepcopy
from typing import Union, List

import inflection
import numpy as np
import pandas as pd
from frozendict import frozendict

from autoflow.constants import VARIABLE_PATTERN
from autoflow.manager.data_container.base import DataContainer
from autoflow.manager.data_container.utils import copy_data_container_structure
from autoflow.utils.dataframe import inverse_dict, process_duplicated_columns
from autoflow.utils.hash import get_hash_of_dataframe, get_hash_of_dict




class DataFrameContainer(DataContainer):
    VALID_INSTANCE = (np.ndarray, pd.DataFrame)

    def __init__(self, dataset_type, dataset_path=None, dataset_instance=None, dataset_id=None, resource_manager=None,
                 dataset_metadata=frozendict()):
        self.column_descriptions = None
        self.feature_groups = pd.Series([])
        self.columns_mapper = {}
        super(DataFrameContainer, self).__init__(dataset_type, dataset_path, dataset_instance, dataset_id,
                                                 resource_manager, dataset_metadata)

    def process_dataset_instance(self, dataset_instance):
        if isinstance(dataset_instance, np.ndarray):
            dataset_instance = pd.DataFrame(dataset_instance,
                                            columns=[f"column_{i}" for i in range(dataset_instance.shape[1])])
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

    def set_feature_groups(self, feature_groups):
        if not isinstance(feature_groups, pd.Series):
            feature_groups = pd.Series(feature_groups)
        assert len(feature_groups) == self.shape[1], "feature_groups' length should equal to features' length."
        self.feature_groups = feature_groups

    def set_column_descriptions(self, column_descriptions):
        assert isinstance(column_descriptions, dict)
        column_descriptions = deepcopy(column_descriptions)
        for feat_grp, item_list in column_descriptions.items():
            if isinstance(item_list, str):
                if item_list not in self.columns:
                    column_descriptions[feat_grp] = []
                    self.logger.info(f"'{item_list}' didn't exist in {self.dataset_type}'s columns, ignore.")
            else:
                ans = []
                for i, elem in enumerate(item_list):
                    if elem in self.columns:
                        ans.append(elem)
                    else:
                        self.logger.info(f"'{elem}' didn't exist in {self.dataset_type}'s columns, ignore.")
                column_descriptions[feat_grp] = ans
        should_pop = []
        for key, value in column_descriptions.items():
            if isinstance(value, list) and len(value) == 0:
                should_pop.append(key)
        for key in should_pop:
            self.logger.info(
                f"After processing, feature_gourp '{key}' is empty, should discard in {self.dataset_type}.")
            column_descriptions.pop(key)
        self.column_descriptions = column_descriptions

    def filter_feature_groups(self, feature_group: Union[List, str], copy=True, isin=True):  # , inplace=False
        if feature_group == "all":  # todo 用正则表达式判断
            feature_group = np.unique(self.feature_groups).tolist()
        # 用于过滤feature_groups
        if isinstance(feature_group, str):
            feature_group = [feature_group]
        result = copy_data_container_structure(self)
        loc = result.feature_groups.isin(feature_group)  # 实际操作的部分
        if not isin:
            loc = (~loc)
        filter_data=self.data.loc[:, self.columns[loc]]
        if copy:
            filter_data=deepcopy(filter_data)
        result.data = filter_data
        result.set_feature_groups(result.feature_groups[loc])
        # 不需要考虑column_descriptions
        return result

    def concat_to(self, other):
        assert isinstance(other, DataFrameContainer)
        new_df = pd.concat([self.data, other.data], axis=1)
        dataset_metadata = deepcopy(self.dataset_metadata)
        dataset_metadata.update(other.dataset_metadata)
        columns, index2newName = process_duplicated_columns(new_df.columns)
        if index2newName:
            new_df.columns = columns
            new_index2newName = deepcopy(dataset_metadata.get("index2newName", {}))
            new_index2newName.update(index2newName)
            dataset_metadata.update({"index2newName": new_index2newName})
        new_feature_groups = pd.concat([self.feature_groups, other.feature_groups], ignore_index=True)
        result = DataFrameContainer(self.dataset_type, dataset_instance=new_df,
                                    resource_manager=self.resource_manager,
                                    dataset_metadata=dataset_metadata)
        result.set_feature_groups(new_feature_groups)
        # 不需要考虑column_descriptions
        return result

    def replace_feature_groups(self, old_feature_group: Union[List[str], str],
                               values: Union[np.ndarray, pd.DataFrame],
                               new_feature_group: Union[str, List[str], pd.Series]):
        if old_feature_group == "all":
            old_feature_group = np.unique(self.feature_groups).tolist()
        if isinstance(old_feature_group, str):
            old_feature_group = [old_feature_group]

        # 将 new_feature_groups 从str表达为list # fixme 类似[num]的情况， 会触发 assert
        if isinstance(new_feature_group, str):
            new_feature_group = [new_feature_group] * values.shape[1]
        assert len(new_feature_group) == values.shape[1]
        new_feature_group = pd.Series(new_feature_group)

        # 实值是ndarray的情况
        if isinstance(values, np.ndarray):
            # new_df 的 columns
            replaced_columns = self.columns[self.feature_groups.isin(old_feature_group)]
            if len(replaced_columns) == values.shape[1]:
                columns = replaced_columns
            else:
                # 特征数发生改变，一共有一对多，多对多，多对一三种情况
                # todo: 这里先采用简单的实现方法，期待新的解决方法
                columns = [f"{x}_{i}" for i, x in enumerate(new_feature_group)]
            values = pd.DataFrame(values, columns=columns)

        deleted_df = self.filter_feature_groups(old_feature_group, True, False)
        new_df = DataFrameContainer(self.dataset_type, dataset_instance=values,
                                    resource_manager=self.resource_manager,
                                    dataset_metadata=self.dataset_metadata)
        new_df.set_feature_groups(new_feature_group)
        new_df.index = deleted_df.index
        return deleted_df.concat_to(new_df)

    def sub_sample(self, index):
        new_df = copy_data_container_structure(self)
        new_df.data = deepcopy(self.data.iloc[index, :])
        return new_df

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
        return super(DataFrameContainer, self).__str__() + f"\nfeature_groups: {list(self.feature_groups)}"

    def __repr__(self):
        return super(DataFrameContainer, self).__repr__() + f"\nfeature_groups: {list(self.feature_groups)}"

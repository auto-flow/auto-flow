#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import hashlib
from collections import defaultdict
from copy import deepcopy
from typing import Union, List

import inflection
import numpy as np
import pandas as pd
from frozendict import frozendict

from autoflow.constants import VARIABLE_PATTERN, UNIQUE_FEATURE_GROUPS
from autoflow.data_container.base import DataContainer
from autoflow.utils.dataframe import inverse_dict, process_duplicated_columns, get_unique_col_name
from autoflow.utils.hash import get_hash_of_dataframe, get_hash_of_dict, get_hash_of_str


class DataFrameContainer(DataContainer):
    VALID_INSTANCE = (np.ndarray, pd.DataFrame)
    dataset_type = "dataframe"

    def __init__(self, dataset_source="", dataset_path=None, dataset_instance=None, dataset_id=None,
                 resource_manager=None,
                 dataset_metadata=frozendict()):
        self.column_descriptions = None
        self.feature_groups = pd.Series([])
        self.columns_mapper = {}
        super(DataFrameContainer, self).__init__(dataset_source, dataset_path, dataset_instance, dataset_id,
                                                 resource_manager, dataset_metadata)

    def process_dataset_instance(self, dataset_instance):
        if isinstance(dataset_instance, np.ndarray):
            dataset_instance = pd.DataFrame(dataset_instance,
                                            columns=[f"column_{i}" for i in range(dataset_instance.shape[1])])
        elif isinstance(dataset_instance, pd.DataFrame):
            origin_columns = dataset_instance.columns
            columns = pd.Series(origin_columns)
            # 1. remove duplicated columns
            unique_col, counts = np.unique(columns, return_counts=True)
            dup_cols = unique_col[counts > 1]
            if len(dup_cols) > 0:
                self.logger.warning(f"Column {dup_cols.tolist()} are duplicated!")
                for dup_col in dup_cols:
                    # in this loop target is eliminate duplicated column `dup_col`
                    while np.sum(columns == dup_col) >= 1:
                        # 1. find first position `dup_col` appear
                        first_ix = columns.tolist().index(dup_col)  # todo: 更好的办法
                        # 2. replace
                        columns[first_ix] = get_unique_col_name(columns, dup_col)
            # set unique columns to dataset_instance
            dataset_instance.columns = columns
            # 2. rename dirty columns
            for i, column in enumerate(columns):
                if not VARIABLE_PATTERN.match(column):
                    columns[i] = get_unique_col_name(columns, "col")
            # 3. adapt to database standard
            columns = pd.Series([inflection.underscore(column).lower() for column in columns])
            # fixme: 可能出现 NAME name 的情况导致重名
            # 4. assemble columns_mapper
            self.columns_mapper = dict(zip(origin_columns, columns))
        else:
            raise NotImplementedError
        return dataset_instance

    def get_hash(self):
        assert self.column_descriptions is not None
        m = hashlib.md5()
        get_hash_of_str(self.dataset_type, m)
        get_hash_of_str(self.dataset_source, m)
        get_hash_of_str(str(list(self.columns)), m)
        get_hash_of_dict(self.column_descriptions, m)
        return get_hash_of_dataframe(self.data, m)

    def read_local(self, path: str):
        if path.endswith(".csv"):
            return pd.read_csv(path)
        else:
            raise NotImplementedError

    def upload(self, upload_type="fs"):
        assert upload_type in ("table", "fs")
        self.dataset_id = self.get_hash()
        if self.dataset_id == self.uploaded_hash:
            return
        dataset_path = self.resource_manager.get_dataset_path(self.dataset_id)
        dataset_path = self.resource_manager.upload_df_to_fs(self.data, dataset_path)
        response = self.resource_manager.insert_dataset_record(
            self.dataset_id, self.dataset_metadata, self.dataset_type, dataset_path,
            upload_type, self.dataset_source, self.column_descriptions,
            self.columns_mapper, list(self.columns)
        )
        if response["length"] == 0:
            self.logger.info(f"Dataset ID: {self.dataset_id} is already exists, "
                             f"{self.dataset_source} will not upload. ")
        super(DataFrameContainer, self).upload(upload_type)

    def download(self, dataset_id):
        records = self.resource_manager.get_dataset_records(dataset_id)
        if len(records) == 0:
            raise ValueError(f"dataset_id: {dataset_id} didn't exists.")
        record = records[0]
        column_descriptions = record["column_descriptions"]
        self.columns_mapper = record["columns_mapper"]
        self.dataset_source = record["dataset_source"]
        self.dataset_metadata = record["dataset_metadata"]
        dataset_path = record["dataset_path"]
        upload_type = record["upload_type"]
        columns = record["columns"]
        if upload_type == "table":
            df = self.resource_manager.download_df_from_table(dataset_id, columns, self.columns_mapper)
        else:
            df = self.resource_manager.download_df_from_fs(dataset_path, columns)
        # inverse_columns_mapper = inverse_dict(self.columns_mapper)
        # df.columns.map(inverse_columns_mapper)
        # todo: 建立本地缓存，防止二次下载
        self.data = df
        self.set_column_descriptions(column_descriptions)

    def set_feature_groups(self, feature_groups):
        if not isinstance(feature_groups, pd.Series):
            feature_groups = pd.Series(feature_groups)
        feature_groups.index = range(len(feature_groups))
        assert len(feature_groups) == self.shape[1], "feature_groups' length should equal to features' length."
        self.feature_groups = feature_groups
        self.column_descriptions = self.feature_groups2column_descriptions(feature_groups)

    def feature_groups2column_descriptions(self, feature_groups):
        result = defaultdict(list)
        for column, feature_group in zip(self.columns, feature_groups):
            if feature_group in UNIQUE_FEATURE_GROUPS:
                result[feature_group] = column
            else:
                result[feature_group].append(column)
        return dict(result)

    def column_descriptions2feature_groups(self, column_descriptions):
        column2feat_grp = {}
        for feat_grp, columns in column_descriptions.items():
            if isinstance(columns, list):
                for column in columns:
                    column2feat_grp[column] = feat_grp
            else:
                column2feat_grp[columns] = feat_grp
        feature_groups = pd.Series(self.columns)
        feature_groups = feature_groups.map(column2feat_grp)
        return feature_groups

    def set_column_descriptions(self, column_descriptions):
        assert isinstance(column_descriptions, dict)
        column_descriptions = deepcopy(column_descriptions)
        for feat_grp, item_list in column_descriptions.items():
            if isinstance(item_list, str):
                if item_list not in self.columns:
                    column_descriptions[feat_grp] = []
                    self.logger.debug(f"'{item_list}' didn't exist in {self.dataset_source}'s columns, ignore.")
            else:
                ans = []
                for i, elem in enumerate(item_list):
                    if elem in self.columns:
                        ans.append(elem)
                    else:
                        self.logger.debug(f"'{elem}' didn't exist in {self.dataset_source}'s columns, ignore.")
                column_descriptions[feat_grp] = ans
        should_pop = []
        for key, value in column_descriptions.items():
            if isinstance(value, list) and len(value) == 0:
                should_pop.append(key)
        for key in should_pop:
            self.logger.debug(
                f"After processing, feature_gourp '{key}' is empty, should discard in {self.dataset_source}.")
            column_descriptions.pop(key)

        self.column_descriptions = column_descriptions
        self.feature_groups = self.column_descriptions2feature_groups(column_descriptions)

    def filter_feature_groups(self, feature_group: Union[List, str], copy=True, isin=True):  # , inplace=False
        if feature_group is None:
            return self
        if feature_group == "all":  # todo 用正则表达式判断
            feature_group = np.unique(self.feature_groups).tolist()
        # 用于过滤feature_groups
        if isinstance(feature_group, str):
            feature_group = [feature_group]
        result = self.copy()
        assert feature_group is not None
        loc = result.feature_groups.isin(feature_group)  # 实际操作的部分
        if not isin:
            loc = (~loc)
        filter_data = self.data.loc[:, self.columns[loc]]
        if copy:
            filter_data = deepcopy(filter_data)
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
        result = DataFrameContainer(self.dataset_source, dataset_instance=new_df,
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
        if new_feature_group is None:
            self.logger.debug("new_feature_group is None, return all feature_groups")
            assert values.shape[1]==self.shape[1]
            assert isinstance(values, pd.DataFrame)
            result=self.copy()
            result.data=values
            return result
        if isinstance(old_feature_group, str):
            old_feature_group = [old_feature_group]

        # 将 new_feature_groups 从str表达为list # fixme 类似[num]的情况， 会触发 assert
        # fixme: 有可能出现列名重复的情况
        if isinstance(new_feature_group, str):
            new_feature_group = [new_feature_group] * values.shape[1]
        assert len(new_feature_group) == values.shape[1]
        new_feature_group = pd.Series(new_feature_group)

        # 实值是ndarray的情况
        if isinstance(values, np.ndarray):
            # new_df 的 columns
            replaced_columns = self.columns[self.feature_groups.isin(old_feature_group)]
            index = self.index
            if len(replaced_columns) == values.shape[1]:
                columns = replaced_columns
            else:
                # 特征数发生改变，一共有一对多，多对多，多对一三种情况
                # todo: 这里先采用简单的实现方法，期待新的解决方法
                columns = [f"{x}_{i}" for i, x in enumerate(new_feature_group)]
            values = pd.DataFrame(values, columns=columns, index=index)

        deleted_df = self.filter_feature_groups(old_feature_group, True, False)
        new_df = DataFrameContainer(self.dataset_source, dataset_instance=values,
                                    resource_manager=self.resource_manager,
                                    dataset_metadata=self.dataset_metadata)
        new_df.set_feature_groups(new_feature_group)
        new_df.index = deleted_df.index  # 非常重要的一步
        return deleted_df.concat_to(new_df)

    def sub_sample(self, index):
        new_df = self.copy()
        # fixme
        new_df.data = deepcopy(new_df.data.iloc[index, :])
        return new_df

    def sub_feature(self, index):
        new_df = self.copy()
        if isinstance(index, np.ndarray) and index.dtype == int:
            new_df.data = deepcopy(new_df.data.iloc[:, index])
            new_df.set_feature_groups(new_df.feature_groups[index])
        else:
            # todo: 如果index是列名序列
            raise NotImplementedError
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

    @property
    def feature_groups_str(self):
        if len(self.feature_groups) < 20:
            return str(list(self.feature_groups))
        return str(self.feature_groups)

    def __str__(self):
        return super(DataFrameContainer, self).__str__() + f"\nfeature_groups: {self.feature_groups_str}"

    def __repr__(self):
        return super(DataFrameContainer, self).__repr__() + f"\nfeature_groups: {self.feature_groups_str}"

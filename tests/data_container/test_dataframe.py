#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
import pandas as pd
from pandas import Index

from autoflow.data_container import DataFrameContainer
from autoflow.datasets import load
from autoflow.test.base import LocalResourceTestCase


class TestDataFrameContainer(LocalResourceTestCase):
    def test_upload_download(self):
        titanic_df = load("titanic")
        titanic_df.index = reversed(titanic_df.index)
        dc = DataFrameContainer(dataset_instance=titanic_df, resource_manager=self.mock_resource_manager)
        feat_grp = [f"feat_{i}" for i in range(dc.shape[1])]
        dc.set_feature_groups(feat_grp)
        column_descriptions = dc.column_descriptions
        dc.upload()
        dataset_id = dc.dataset_hash
        download_dc = DataFrameContainer("Unittest", dataset_id=dataset_id, resource_manager=self.mock_resource_manager)
        self.assertTrue(np.all(download_dc.data.fillna(0) == dc.data.fillna(0)))
        self.assertTrue(np.all(download_dc.feature_groups == dc.feature_groups))
        self.assertTrue(np.all(download_dc.columns == dc.columns))
        self.assertTrue(np.all(download_dc.index == dc.index))
        self.assertEqual(download_dc.column_descriptions, dc.column_descriptions)
        self.assertEqual(download_dc.columns_mapper, dc.columns_mapper)
        self.assertEqual(download_dc.dataset_type, dc.dataset_type)
        self.assertEqual(download_dc.dataset_source, dc.dataset_source)

    def test_set_dirty_columns(self):
        titanic_df = load("titanic")
        columns = pd.Series(titanic_df.columns)
        columns = ["@"] * len(columns)
        titanic_df.columns = columns
        dc = DataFrameContainer(dataset_instance=titanic_df, resource_manager=self.mock_resource_manager)
        wanted_columns = Index(['col', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7',
                                'col_8', 'col_9', 'col_10', 'col_11'],
                               dtype='object')
        self.assertTrue(np.all(dc.columns == wanted_columns))

    def test_set_same_column(self):
        titanic_df = load("titanic")
        columns = pd.Series(titanic_df.columns)
        columns = ["@"] * len(columns)
        columns[1] = "same"
        columns[2] = "same"
        columns[3] = "same"
        columns[5] = "ok"
        columns[6] = "ok"
        titanic_df.columns = columns
        dc = DataFrameContainer(dataset_instance=titanic_df, resource_manager=self.mock_resource_manager)
        wanted=Index(['col', 'same_1', 'same_2', 'same_3', 'col_1', 'ok_1', 'ok_2', 'col_2',
               'col_3', 'col_4', 'col_5', 'col_6'],
              dtype='object')
        self.assertTrue(np.all(dc.columns==wanted))
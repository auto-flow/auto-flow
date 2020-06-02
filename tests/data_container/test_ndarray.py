#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
import pandas as pd

from autoflow import NdArrayContainer
from autoflow.tests.base import LocalResourceTestCase


class TestDataFrameContainer(LocalResourceTestCase):
    def test_instancing(self):
        dc = NdArrayContainer(dataset_instance=[1, 2, 3], resource_manager=self.mock_resource_manager)
        self.assertTrue(np.all(dc.data == np.array([1, 2, 3])))
        dc = NdArrayContainer(dataset_instance=pd.Series([1, 2, 3]), resource_manager=self.mock_resource_manager)
        self.assertTrue(np.all(dc.data == np.array([1, 2, 3])))

    def test_upload_download(self):
        in_data = [1, 2, 3, 4, 5]
        dc = NdArrayContainer(dataset_instance=in_data, resource_manager=self.mock_resource_manager)
        dc.upload()
        d_dc = NdArrayContainer(dataset_id=dc.dataset_hash, resource_manager=self.mock_resource_manager)
        self.assertTrue(np.all(d_dc.data == np.array(in_data)))

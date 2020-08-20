#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import unittest
import numpy as np
import h5py
from autoflow import datasets
from autoflow.utils.hash import get_hash_of_dataframe_csv, get_hash_of_dict, get_hash_of_array


class TestDict(unittest.TestCase):

    def test_get_hash_of_dataframe_csv(self):
        train_df = datasets.load("titanic")
        hash_value1 = get_hash_of_dataframe_csv(train_df, L=51)
        hash_value2 = get_hash_of_dataframe_csv(train_df, L=100)
        self.assertTrue(hash_value1 == hash_value2)

    def test_get_hash_of_dict(self):
        o1 = {
            "A": [3, 4, 2],
            "C": {
                "D": [3, 2, 1, "s", False],
                "B": "9"
            },
            "B": []
        }
        o2 = {
            "A": [4, 3, 2],
            "B": [],
            "C": {
                "B": "9",
                "D": [False, 3, 2, 1, "s"],
            },
        }
        o3 = {
            "A": [4, 3, 1],
            "B": [],
            "C": {
                "B": "9",
                "D": [False, 3, 2, 1, "s"],
            },
        }
        self.assertEqual(get_hash_of_dict(o1), get_hash_of_dict(o2))
        self.assertEqual(get_hash_of_dict(o1), get_hash_of_dict(o1))
        self.assertNotEqual(get_hash_of_dict(o1), get_hash_of_dict(o3))
        self.assertNotEqual(get_hash_of_dict(o2), get_hash_of_dict(o3))

    def test_hash_arr_hdf5(series):
        np.random.seed(0)
        data_to_write = np.random.random(size=(100, 100))
        hv1 = get_hash_of_array(data_to_write)
        print(hv1)
        with h5py.File('name-of-file.h5', 'w') as hf:
            hf.create_dataset("name-of-dataset", data=data_to_write)

        with h5py.File('name-of-file.h5', 'r') as hf:
            data = hf['name-of-dataset'][:]

        hv2 = get_hash_of_array(data)
        print(hv2)
        assert hv1 == hv2


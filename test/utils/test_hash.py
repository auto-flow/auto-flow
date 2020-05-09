#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import unittest
from pathlib import Path

import pandas as pd

import autoflow
from autoflow.utils.hash import get_hash_of_dataframe


class TestDict(unittest.TestCase):

    def test_get_hash_of_dataframe(self):
        examples_path = Path(autoflow.__file__).parent.parent / "examples"
        train_df = pd.read_csv(examples_path / "data/train_classification.csv")
        hash_value1 = get_hash_of_dataframe(train_df, L=51)
        hash_value2 = get_hash_of_dataframe(train_df, L=100)
        self.assertTrue(hash_value1 == hash_value2)

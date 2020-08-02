#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
import pandas as pd
import unittest

from autoflow.feature_engineer.encode import CombineRare

class TestCombineRare(unittest.TestCase):
    def test_combine_rare(self):
        a1 = np.arange(100, dtype=int)
        a1[:10] = 0
        a2 = np.arange(100, dtype=object)
        a2[:10] = "A"
        df = pd.DataFrame({"col1": a1, "col2": a2, "col3": a2})
        df["col4"] = pd.Categorical(["D"] * 25 + ["C"] * 25 + ["B"] * 25 + ["A"] * 24 + ["Z"],categories=["Z","D","C","B","A"])
        df['col3'] = df['col3'].astype('category')
        df["col5"]=range(100)
        combine_rare = CombineRare()
        df_t = combine_rare.fit_transform(df)
        print(df_t)
        assert df_t.dtypes[2].name == "category"
        assert np.all(df_t.col3.cat.categories == pd.Series(['A', 'Others Infrequent']))
        assert np.all(df_t["col4"].cat.categories==pd.Series( ['D', 'C', 'B', 'A', 'Others Infrequent']))
        assert "col5" not in df_t.columns

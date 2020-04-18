#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import hashlib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from autoflow.manager.data_manager import DataManager
import autoflow
from autoflow.utils.hash import get_hash_of_Xy, get_hash_of_str

examples_path = Path(autoflow.__file__).parent.parent / "examples"
train_df = pd.read_csv(examples_path/"data/train_classification.csv")
# ss = ShuffleSplit(n_splits=1, random_state=0, test_size=0.25)
# train_ix, test_ix = next(ss.split(df))
# df_train = df.iloc[train_ix, :]
# df_test = df.iloc[test_ix, :]
column_descriptions={
    "target": "Survived"
}
data_manager=DataManager(X_train=df_train,X_test=df_test,column_descriptions=column_descriptions)
m=hashlib.md5()
get_hash_of_Xy(data_manager.X_train,data_manager.y_train,m)
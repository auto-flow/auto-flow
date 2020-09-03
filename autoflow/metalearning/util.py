#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.feature_engineer.impute import SimpleImputer
from autoflow.feature_engineer.encode import CombineRare
from sklearn.pipeline import Pipeline
from category_encoders import OneHotEncoder

def get_data_preprocessor(cols):
    data_preprocessor=Pipeline([
        ("imputer",SimpleImputer(categorical_feature=cols, missing_rate=1,inclusive=False)),
        ("combine_rare",CombineRare(minimum_fraction=0.1,categorical_feature=cols, drop_invariant=False)),
        ("encoder", OneHotEncoder(cols=cols, drop_invariant=True))
    ])
    return data_preprocessor
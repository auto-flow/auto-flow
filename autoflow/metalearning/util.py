#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json

import requests
from category_encoders import OneHotEncoder
from sklearn.pipeline import Pipeline

from autoflow.feature_engineer.encode import CombineRare
from autoflow.feature_engineer.impute import SimpleImputer


def get_data_preprocessor(cols):
    data_preprocessor = Pipeline([
        ("imputer", SimpleImputer(categorical_feature=cols, missing_rate=1, inclusive=False)),
        ("combine_rare", CombineRare(minimum_fraction=0.1, categorical_feature=cols, drop_invariant=False)),
        ("encoder", OneHotEncoder(cols=cols, drop_invariant=True))
    ])
    return data_preprocessor


def load_metalearning_repository(fs, mtl_repo_path, name):
    json_path = fs.join(mtl_repo_path, f"{name}.json")
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json',
    }
    if not fs.exists(json_path):
        url = f"https://autoflow.s3-ap-southeast-1.amazonaws.com/{name}.json"
        data = requests.get(url, headers=headers).json()
        with open(json_path, "w+") as f:
            json.dump(data, f)
    else:
        with open(json_path, "r") as f:
            data = json.load(f)
    return data

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from pathlib import Path

import joblib
import numpy as np
import openml  # todo : lazy import
import pandas as pd

__all__ = ["load"]


def get_file(data_dir, path_list):
    for file_name in path_list:
        for mode in ["csv", "bz2"]:
            path = (data_dir / f"{file_name}.{mode}")
            if path.exists():
                if mode == "csv":
                    return pd.read_csv(path)
                elif mode == "bz2":
                    return joblib.load(path)
                else:
                    return None
    return None


def load(name, return_train_test=False):
    data_dir = Path(__file__).parent / "data"
    train_set = get_file(data_dir, [name, f"{name}_train"])
    assert train_set is not None
    if return_train_test:
        test_set = get_file(data_dir, [f"{name}_test"])
        assert test_set is not None
        return train_set, test_set
    return train_set


def load_task(task_id):
    task = openml.tasks.get_task(task_id)
    train_indices, test_indices = task.get_train_test_split_indices()
    dataset = openml.datasets.get_dataset(task.dataset_id)
    df, y, cat, _ = dataset.get_data(target=task.target_name)
    y = np.array(y)
    X_train = df.iloc[train_indices, :]
    y_train = y[train_indices]
    X_test = df.iloc[test_indices, :]
    y_test = y[test_indices]

    del _
    del dataset

    return X_train, y_train, X_test, y_test, cat

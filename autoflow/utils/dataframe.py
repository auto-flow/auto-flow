from copy import deepcopy
from typing import Optional, List, Union

import numpy as np
import pandas as pd


def process_dataframe(X: Union[pd.DataFrame, np.ndarray], copy=True) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        if copy:
            X_ = deepcopy(X)
        else:
            X_ = X
    elif isinstance(X, np.ndarray):
        X_ = pd.DataFrame(X, columns=range(X.shape[1]))
    else:
        raise NotImplementedError
    return X_


def replace_nan_to_None(df: pd.DataFrame) -> pd.DataFrame:
    df = deepcopy(df)
    for column, dtype in zip(df.columns, df.dtypes):
        if dtype == object:
            df[column] = df[column].apply(lambda x: None if pd.isna(x) else x)
    return df


def pop_if_exists(df: pd.DataFrame, col: str) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    if col in df.columns:
        return df.pop(col)
    else:
        return None


def replace_dict(dict_: dict, from_, to_):
    for k, v in dict_.items():
        if v == from_:
            dict_[k] = to_


def replace_dicts(dicts, from_, to_):
    for dict_ in dicts:
        replace_dict(dict_, from_, to_)


def get_unique_col_name(columns: List[str], wanted: str):
    while wanted in columns:
        wanted = wanted + "_"
    return wanted


def inverse_dict(dict_: dict):
    dict_ = deepcopy(dict_)
    return {v: k for k, v in dict_.items()}


def rectify_dtypes(df: pd.DataFrame):
    # make sure: only (str, int, float, bool) is valid
    object_columns = get_object_columns(df)
    for object_column in object_columns:
        if not np.any(df[object_column].apply(lambda x: isinstance(x, str))):
            if np.any(df[object_column].apply(lambda x: isinstance(x, float))):
                df[object_column] = df[object_column].astype(float)
            else:
                df[object_column] = df[object_column].astype(int)


def get_object_columns(df_: pd.DataFrame) -> List[str]:
    return list(df_.dtypes[df_.dtypes == object].index)


class DataFrameValuesWrapper():
    def __init__(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            self.array = X.values
            self.dataframe = X
            self.origin = "dataframe"
        elif isinstance(X, np.ndarray):
            self.array = X
            self.dataframe = pd.DataFrame(X)
            self.origin = "array"

    def wrap_to_dataframe(self, array):
        return pd.DataFrame(array, columns=self.dataframe.columns, index=self.dataframe.index)

from copy import deepcopy
from typing import List, Union, Tuple, Dict

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


def replace_dict(dict_: dict, from_, to_):
    for k, v in dict_.items():
        if v == from_:
            dict_[k] = to_


def replace_dicts(dicts, from_, to_):
    for dict_ in dicts:
        replace_dict(dict_, from_, to_)


def get_unique_col_name(columns: Union[pd.Index, pd.Series], wanted: str):
    cnt = 1
    while np.sum(columns == wanted) >= 1:
        ix = wanted.rfind("_")
        if ix<0:
            ix=len(wanted)
        wanted = wanted[:ix] + f"_{cnt}"
        cnt += 1
    return wanted


def process_duplicated_columns(columns: pd.Index) -> Tuple[pd.Index, Dict[str, str]]:
    # 查看是否有重复列，并去重
    if isinstance(columns, pd.Index):
        columns = pd.Series(columns)
    else:
        columns = deepcopy(columns)
    unq, cnt = np.unique(columns, return_counts=True)
    if len(unq) == len(columns):
        return columns, {}
    duplicated_columns = unq[cnt > 1]
    index2newName = {}
    for duplicated_column in duplicated_columns:
        for ix in reversed(np.where(columns == duplicated_column)[0]):
            new_name = get_unique_col_name(columns, columns[ix])
            columns[ix] = new_name
            index2newName[ix] = new_name
    assert len(np.unique(columns)) == len(columns)
    return columns, index2newName


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

# if __name__ == '__main__':
#     columns = pd.Index([str(x) for x in [1, 2, 3, 2, 3, 4, 5]])
#     columns, index2newName = process_duplicated_columns(columns)
#     print(columns)
#     print(index2newName)

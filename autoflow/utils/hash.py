import hashlib
from copy import deepcopy
from math import ceil
from typing import Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from autoflow.utils.dataframe import get_object_columns
from autoflow.utils.dict_ import sort_dict


def get_hash_of_file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_hash_of_array(X, m=None):
    if m is None:
        m = hashlib.md5()

    if issparse(X):
        m.update(X.indices)
        m.update(X.indptr)
        m.update(X.data)
        m.update(str(X.shape).encode('utf8'))
    else:
        if X.flags['C_CONTIGUOUS']:
            m.update(X.data)
            m.update(str(X.shape).encode('utf8'))
        else:
            X_tmp = np.ascontiguousarray(X.T)
            m.update(X_tmp.data)
            m.update(str(X_tmp.shape).encode('utf8'))

    hash = m.hexdigest()
    return hash


def get_hash_of_dict(dict_, m=None):
    if m is None:
        m = hashlib.md5()

    sorted_dict = sort_dict(deepcopy(dict_))
    # sorted_dict = deepcopy(dict_)
    m.update(str(sorted_dict).encode("utf-8"))
    return m.hexdigest()


def get_hash_decimal_of_str(x, m=None):
    if m is None:
        m = hashlib.md5()
    if isinstance(x, str):
        hash_str = get_hash_of_str(x, m)
        hash_int = int(hash_str[:5], 16)
        return hash_int
    elif isinstance(x, bool):
        return int(x)
    elif isinstance(x, (int, float)):
        return x
    else:
        raise NotImplementedError


def get_hash_of_dataframe_deprecated(df: pd.DataFrame, m=None):
    if m is None:
        m = hashlib.md5()
    df_ = deepcopy(df)
    object_columns = get_object_columns(df_)
    for objest_column in object_columns:
        df_[objest_column] = df_[objest_column].apply(get_hash_decimal_of_str)  # .astype("float")
    df_.sort_index(axis=0, inplace=True)
    df_.sort_index(axis=1, inplace=True)
    return get_hash_of_array(df_.values, m)


def get_hash_of_dataframe_csv(df: pd.DataFrame, m=None, L=500):
    if m is None:
        m = hashlib.md5()
    sp0 = df.shape[0]
    N = ceil(sp0 / L)
    result = ""
    s = df.iloc[:0, :].to_csv(float_format="%.3f", index=False).encode()
    get_hash_of_str(s, m)
    for i in range(N):
        s = df.iloc[i * L:min(sp0, (i + 1) * L)].to_csv(float_format="%.3f", index=False, header=None).encode()
        result = get_hash_of_str(s, m)
    return result


def get_hash_of_dataframe(df: pd.DataFrame, m=None, L=500):
    if m is None:
        m = hashlib.md5()
    eq_obj = (df.dtypes == object)
    if np.any(eq_obj):
        get_hash_of_dataframe_csv(df.select_dtypes(include=object), m, L)
        result = get_hash_of_array(df.select_dtypes(exclude=object).values, m)
    else:
        result = get_hash_of_array(df.values, m)

    return result


def get_hash_of_Xy(X: Union[pd.DataFrame, np.ndarray, None],
                   y: Union[pd.DataFrame, np.ndarray, pd.Series, None] = None,
                   m=None):
    if X is None:
        return ""
    X = pd.DataFrame(X)
    df = X
    if y is not None:
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if len(y.shape) == 1:
            y = y[:, None]
        y = pd.DataFrame(y, columns=["y"])
        if y.shape[1] != df.shape[1]:
            return get_hash_of_dataframe(df, m)
        df = pd.concat([X, y], ignore_index=True)
    return get_hash_of_dataframe(df, m)


def get_hash_of_str(s: Union[str, bytes], m=None):
    if m is None:
        m = hashlib.md5()
    if isinstance(s, str):
        s = s.encode("utf-8")
    m.update(s)
    return m.hexdigest()

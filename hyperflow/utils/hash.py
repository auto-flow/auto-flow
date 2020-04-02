import hashlib
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from hyperflow.utils.dict import sort_dict


def get_hash_of_array(X):
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


def get_hash_of_dict(dict_):
    m = hashlib.md5()
    m.update(str(sort_dict(deepcopy(dict_))).encode("utf-8"))
    return m.hexdigest()


def get_hash_of_dataframe(df: pd.DataFrame):
    df_ = deepcopy(df)
    df_.sort_index(axis=0, inplace=True)
    df_.sort_index(axis=1, inplace=True)
    return get_hash_of_array(df_.values)


def get_hash_of_Xy(X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray, pd.Series, None] = None):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    df = X
    if y is not None:
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if len(y.shape) == 1:
            y = y[:, None]
        y=pd.DataFrame(y,columns=["y"])
        if y.shape[1]!=df.shape[1]:
            return get_hash_of_dataframe(df)
        df=pd.concat([X,y],ignore_index=True)
    return get_hash_of_dataframe(df)
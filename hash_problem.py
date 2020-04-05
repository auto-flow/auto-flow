import hashlib
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse


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


def get_hash_decimal_of_str(x):
    if isinstance(x, str):
        hash_str = get_hash_of_str(x)
        hash_int = int(hash_str[:5], 16)
        return hash_int
    elif isinstance(x, bool):
        return int(x)
    elif isinstance(x, (int, float)):
        return x
    else:
        raise NotImplementedError


def get_hash_of_dataframe(df: pd.DataFrame, m=None):
    df_ = deepcopy(df)
    obj_columns = list(df_.dtypes[df_.dtypes == object].index)
    for obj_column in obj_columns:
        df_[obj_column] = df_[obj_column].apply(get_hash_decimal_of_str)  # .astype("float")
    df_.sort_index(axis=0, inplace=True)
    df_.sort_index(axis=1, inplace=True)
    return get_hash_of_array(df_.values, m)


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
            return get_hash_of_dataframe(df)
        df = pd.concat([X, y], ignore_index=True)
    return get_hash_of_dataframe(df, m)


def get_hash_of_str(s: str, m=None):
    if m is None:
        m = hashlib.md5()
    m.update(s.encode("utf-8"))
    return m.hexdigest()


if __name__ == '__main__':
    from hyperflow import XYDataManager
    import pandas as pd
    df=pd.read_csv("../examples/classification/train_classification.csv")
    column_descriptions={
        "id":"PassengerId",
        "target":"Survived"
    }
    data_manager=XYDataManager(X=df,column_descriptions=column_descriptions)
    hash_value=get_hash_of_Xy(data_manager.X_train,data_manager.y_train)
    print(hash_value)


from typing import Optional, List, Union

import numpy as np
import pandas as pd


def pop_if_exists(df: pd.DataFrame, col: str) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    if col in df.columns:
        return df.pop(col)
    else:
        return None


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



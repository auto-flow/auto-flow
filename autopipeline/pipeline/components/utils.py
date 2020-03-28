import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target


def stack_Xs(X_train, X_valid=None, X_test=None):
    Xs = [X_train]
    if X_valid is not None:
        Xs.append(X_valid)
    if X_test is not None:
        Xs.append(X_test)
    if isinstance(Xs[0], pd.DataFrame):
        df = pd.concat(Xs, axis=0)
        df.sort_index(inplace=True)
        return df
    elif isinstance(Xs[0], np.ndarray):
        return np.vstack(Xs)


def get_categorical_features_indices(X, origin_grp):
    if isinstance(X, pd.DataFrame):
        X = X.values
    categorical_features_indices = []
    for i, elem in enumerate(origin_grp):
        if "cat" in elem:
            col = X[:, i]
            try:
                col = col.astype("float")
            except Exception:
                pass
            if type_of_target(col) not in ("binary", "continuous"):  # todo: debug
                categorical_features_indices.append(i)
    return categorical_features_indices

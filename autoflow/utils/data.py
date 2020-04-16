# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.sparse import issparse


def convert_to_num(Ybin):
    """
    Convert binary targets to numeric vector
    typically classification target values
    :param Ybin:
    :return:
    """
    result = np.array(Ybin)
    if len(Ybin.shape) != 1:
        result = np.dot(Ybin, range(Ybin.shape[1]))
    return result


def convert_to_bin(Ycont, nval, verbose=True):
    # Convert numeric vector to binary (typically classification target values)
    if verbose:
        pass
    Ybin = [[0] * nval for x in range(len(Ycont))]
    for i in range(len(Ybin)):
        line = Ybin[i]
        line[np.int(Ycont[i])] = 1
        Ybin[i] = line
    return Ybin


def predict_RAM_usage(X, categorical):
    # Return estimated RAM usage of dataset after OneHotEncoding in bytes.
    estimated_columns = 0
    for i, cat in enumerate(categorical):
        if cat:
            unique_values = np.unique(X[:, i])
            num_unique_values = np.sum(np.isfinite(unique_values))
            estimated_columns += num_unique_values
        else:
            estimated_columns += 1
    estimated_ram = estimated_columns * X.shape[0] * X.dtype.itemsize
    return estimated_ram


def softmax(df):
    if len(df.shape) == 1:
        df[df > 20] = 20
        df[df < -20] = -20
        ppositive = 1 / (1 + np.exp(-df))
        ppositive[ppositive > 0.999999] = 1
        ppositive[ppositive < 0.0000001] = 0
        return np.transpose(np.array((1 - ppositive, ppositive)))
    else:
        # Compute the Softmax like it is described here:
        # http://www.iro.umontreal.ca/~bengioy/dlbook/numerical.html
        tmp = df - np.max(df, axis=1).reshape((-1, 1))
        tmp = np.exp(tmp)
        return tmp / np.sum(tmp, axis=1).reshape((-1, 1))


def densify(X):
    if X is None:
        return X
    if issparse(X):
        return X.todense().getA()
    else:
        return X


def is_cat(s: pd.Series):
    for elem in s:
        if isinstance(elem, (float, int)):
            continue
        else:
            return True
    return False


def is_highR_nan(s: pd.Series, threshold):
    return (np.count_nonzero(pd.isna(s)) / s.size) > threshold


def is_nan(s: pd.Series):
    return np.any(pd.isna(s))


def to_array(X):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.values
    return X

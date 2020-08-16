# -*- encoding: utf-8 -*-
import multiprocessing as mp
from typing import Union

import numpy as np
import pandas as pd
from datefinder import find_dates
from scipy.sparse import issparse
from sklearn.utils.multiclass import type_of_target


def check_n_jobs(n_jobs):
    cpu_count = mp.cpu_count()
    if n_jobs == 0:
        return 1
    elif n_jobs > 0:
        return min(cpu_count, n_jobs)
    else:
        return max(1, cpu_count + 1 + n_jobs)


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


def is_target_need_label_encode(target_col):
    if is_cat(target_col, True):
        unk = np.unique(target_col)
        wanted = np.arange(len(unk), dtype='int32')
        if not np.all(unk == wanted):
            return True
    return False


def is_cat(s: Union[pd.Series, np.ndarray], consider_ordinal_as_cat):
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    if s.dtype == object:
        for elem in s:
            if isinstance(elem, (float, int)):
                continue
            else:
                return True
        s = s.astype('float32')
    if consider_ordinal_as_cat:
        valid_types = ["multiclass"]
        if consider_ordinal_as_cat in (2, "binary"):
            valid_types += ["binary"]
        s = s.dropna()
        tp = type_of_target(s)
        if tp in valid_types:
            return True
    return False


def finite_array(array):
    """
    Replace NaN and Inf (there should not be any!)
    :param array:
    :return:
    """
    a = np.ravel(array)
    maxi = np.nanmax(a[np.isfinite(a)])
    mini = np.nanmin(a[np.isfinite(a)])
    array[array == float('inf')] = maxi
    array[array == float('-inf')] = mini
    return array


def is_highR_nan(s: pd.Series, threshold):
    return (np.count_nonzero(pd.isna(s)) / s.size) > threshold


def is_highC_cat(s: pd.Series, threshold):
    return (np.unique(s.astype("str")).size) >= threshold


def is_nan(s: pd.Series):
    return np.any(pd.isna(s))


def to_array(X):
    if X is None:
        return X
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.values
    return X


def is_text(s, cat_been_checked=False):
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    if not cat_been_checked:
        if not is_cat(s, consider_ordinal_as_cat=False):
            return False
    s = s.dropna()
    s = s.astype(str)
    s = s.str.split(" ")
    s = s.apply(len)
    return np.all(s >= 2) # todo 参考 AG


def is_date(s, cat_been_checked=False):
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    if not cat_been_checked:
        if not is_cat(s, consider_ordinal_as_cat=False):
            return False
    s = s.dropna()
    s = s.astype(str)  # todo 参考 AG
    return all(bool(list(find_dates(elem, strict=True))) for elem in s)


if __name__ == '__main__':
    print(is_text([
        "hello world",
        "good morning"
        "it is a good day"
    ]))
    print(is_text([
        "hello world",
        "good morning",
        0
    ]))
    print(is_text([
        "hello world",
        "good morning",
        "omg"
    ]))
    print(is_text([
        "hello world",
        "hello world",
        "hello world",
    ]))
    print(is_date([
        '2018',
        '2016',
        '658.2.3'
    ]))
    print(is_date([
        '456',
        '456',
        '256'
    ]))

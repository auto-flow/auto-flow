# -*- encoding: utf-8 -*-
import math
from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.utils.multiclass import type_of_target

from autopipeline.constants import binary_classification_task, multiclass_classification_task, \
    multilabel_classification_task, regression_task


def sanitize_array(array):
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
    mid = (maxi + mini) / 2
    array[np.isnan(array)] = mid
    return array


def get_task_from_y(y):
    y_type = type_of_target(y)
    if y_type == "binary":
        task = binary_classification_task
    elif y_type == "multiclass":
        task = multiclass_classification_task
    elif y_type == "multilabel-indicator":
        task = multilabel_classification_task
    elif y_type == "multiclass-multioutput":
        raise NotImplementedError()
    elif y_type == "continuous":
        task = regression_task
    else:
        raise NotImplementedError()
    return task


def vote_predicts(predicts: List[np.ndarray]):
    probas_arr = np.array(predicts)
    proba = np.average(probas_arr, axis=0)
    return proba


def mean_predicts(predicts: List[np.ndarray]):
    probas_arr = np.array(predicts)
    proba = np.average(probas_arr, axis=0)
    return proba


def binarization(array):
    # Takes a binary-class datafile and turn the max value (positive class)
    # into 1 and the min into 0
    array = np.array(array, dtype=float)  # conversion needed to use np.inf
    if len(np.unique(array)) > 2:
        raise ValueError('The argument must be a binary-class datafile. '
                         '{} classes detected'.format(len(np.unique(array))))

    # manipulation which aims at avoid error in data
    # with for example classes '1' and '2'.
    array[array == np.amax(array)] = np.inf
    array[array == np.amin(array)] = 0
    array[array == np.inf] = 1
    return np.array(array, dtype=int)


def multilabel_to_multiclass(array):
    array = binarization(array)
    return np.array([np.nonzero(array[i, :])[0][0] for i in range(len(array))])


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


def float_gcd(a, b):
    def is_int(x):
        return not bool(int(x) - x)

    base = 1
    while not (is_int(a) and is_int(b)):
        a *= 10
        b *= 10
        base *= 10
    return math.gcd(int(a), int(b)) / base


def is_cat(s: pd.Series):
    for elem in s:
        if isinstance(elem, (float, int)):
            continue
        else:
            return True
    return False


def is_nan(s: pd.Series):
    return np.any(pd.isna(s))


def arraylize(X):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.values
    return X

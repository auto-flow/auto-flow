#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from collections import Counter

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.multiclass import type_of_target


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


def get_stratified_sampling_index(array: np.ndarray, proportion: float, random_state: int):
    array = array.astype("float32")
    if type_of_target(array) == "continuous":
        if len(array) < 10:
            n_bins = 2
        else:
            n_bins = 5
        kbins = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")
        array = kbins.fit_transform(array.reshape(-1, 1)).squeeze()
    assert len(array.shape) == 1
    rng = np.random.RandomState(random_state)
    labels = Counter(array)
    origin_index = np.arange(len(array), dtype='int32')
    results = []
    for label in labels:
        mask = (array == label)
        masked_index = origin_index[mask]
        L = max(1, round(len(masked_index) * proportion))
        samples = rng.choice(masked_index, L, replace=False)
        results.append(samples)
    result = np.hstack(results)
    rng.shuffle(result)
    return result

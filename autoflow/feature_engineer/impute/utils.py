"""Utility Functions"""
# Author: Ashim Bhattarai
# License: BSD 3 clause
from copy import deepcopy
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.utils.multiclass import type_of_target


def process_dataframe(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        X_ = deepcopy(X)
    elif isinstance(X, np.ndarray):
        X_ = pd.DataFrame(X, columns=range(X.shape[1]))
    else:
        raise NotImplementedError
    return X_


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


def masked_euclidean_distances(X, Y=None, squared=False,
                               missing_values="NaN", copy=True):
    """Calculates euclidean distances in the presence of missing values

    Computes the euclidean distance between each pair of samples (rows) in X
    and Y, where Y=X is assumed if Y=None.
    When calculating the distance between a pair of samples, this formulation
    essentially zero-weights feature coordinates with a missing value in either
    sample and scales up the weight of the remaining coordinates:

        dist(x,y) = sqrt(weight * sq. distance from non-missing coordinates)
        where,
        weight = Total # of coordinates / # of non-missing coordinates

    Note that if all the coordinates are missing or if there are no common
    non-missing coordinates then NaN is returned for that pair.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_1, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_2, n_features)

    squared : boolean, optional
        Return squared Euclidean distances.

    missing_values : "NaN" or integer, optional
        Representation of missing value

    copy : boolean, optional
        Make and use a deep copy of X and Y (if Y exists)

    Returns
    -------
    distances : {array}, shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from skimpute.utils import masked_euclidean_distances
    >>> nan = float("NaN")
    >>> X = [[0, 1], [1, nan]]
    >>> # distance between rows of X
    >>> masked_euclidean_distances(X, X)
    array([[0.        , 1.41421356],
           [1.41421356, 0.        ]])

    >>> # get distance to origin
    >>> masked_euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])

    References
    ----------
    * John K. Dixon, "Pattern Recognition with Partly Missing Data",
      IEEE Transactions on Systems, Man, and Cybernetics, Volume: 9, Issue:
      10, pp. 617 - 621, Oct. 1979.
      http://ieeexplore.ieee.org/abstract/document/4310090/

    See also
    --------
    paired_distances : distances betweens pairs of elements of X and Y.
    """
    # Import here to prevent circular import
    from .pairwise_external import _get_mask, check_pairwise_arrays

    # NOTE: force_all_finite=False allows not only NaN but also +/- inf
    X, Y = check_pairwise_arrays(X, Y, accept_sparse=False,
                                 force_all_finite=False, copy=copy)
    if (np.any(np.isinf(X)) or
            (Y is not X and np.any(np.isinf(Y)))):
        raise ValueError(
            "+/- Infinite values are not allowed.")

    # Get missing mask for X and Y.T
    mask_X = _get_mask(X, missing_values)

    YT = Y.T
    mask_YT = mask_X.T if Y is X else _get_mask(YT, missing_values)

    # Check if any rows have only missing value
    if np.any(mask_X.sum(axis=1) == X.shape[1]) \
            or (Y is not X and np.any(mask_YT.sum(axis=0) == Y.shape[1])):
        raise ValueError("One or more rows only contain missing values.")

    # else:
    if missing_values not in ["NaN", np.nan] and (
            np.any(np.isnan(X)) or (Y is not X and np.any(np.isnan(Y)))):
        raise ValueError(
            "NaN values present but missing_value = {0}".format(
                missing_values))

    # Get mask of non-missing values set Y.T's missing to zero.
    # Further, casting the mask to int to be used in formula later.
    not_YT = (~mask_YT).astype(np.int32)
    YT[mask_YT] = 0

    # Get X's mask of non-missing values and set X's missing to zero
    not_X = (~mask_X).astype(np.int32)
    X[mask_X] = 0

    # Calculate distances
    # The following formula derived by:
    # Shreya Bhattarai <shreya.bhattarai@gmail.com>

    distances = (
            (X.shape[1] / (np.dot(not_X, not_YT))) *
            (np.dot(X * X, not_YT) - 2 * (np.dot(X, YT)) +
             np.dot(not_X, YT * YT)))

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances if squared else np.sqrt(distances, out=distances)


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
            s = s.dropna()
            tp = type_of_target(s)
            if tp in ("multiclass",):
                return True
    return False


def parse_cat_col(X_: pd.DataFrame, consider_ordinal_as_cat):
    cat_idx = []
    num_idx = []
    for i, column in enumerate(X_.columns):
        col = X_[column]
        if is_cat(col, consider_ordinal_as_cat):
            cat_idx.append(i)
        else:
            num_idx.append(i)
    return np.array(num_idx), np.array(cat_idx)


def build_encoder(X_: pd.DataFrame, y, cat_idx, passed_encoder, additional_data_list: List[np.ndarray], target_type):
    pd.set_option('mode.chained_assignment', None)
    idx2encoder = {}
    result_additional_data_list = deepcopy(additional_data_list)
    for idx in cat_idx:
        col = X_.values[:, idx]
        valid_mask = ~pd.isna(col)
        masked_col = col[valid_mask]
        if y is not None:
            masked_y = y[valid_mask]
        else:
            masked_y = None
        masked_col = masked_col.reshape(-1, 1).astype(str)
        encoder = clone(passed_encoder)
        encoder.fit(masked_col, masked_y)
        idx2encoder[idx] = encoder
        col[valid_mask] = encoder.transform(masked_col).squeeze()
        X_.iloc[:, idx] = col.astype(target_type)
        for additional_data in result_additional_data_list:
            additional_data[idx] = encoder.transform([[str(additional_data[idx])]])[0][0]
    return idx2encoder, X_, result_additional_data_list


def encode_data(X_: pd.DataFrame, idx2encoder, target_type):
    pd.set_option('mode.chained_assignment', None)
    for idx, encoder in idx2encoder.items():
        col = X_.values[:, idx]
        valid_mask = ~pd.isna(col)
        masked_col = col[valid_mask]
        masked_col = masked_col.reshape(-1, 1).astype(str)
        col[valid_mask] = encoder.transform(masked_col).squeeze()
        X_.iloc[:, idx] = col
    return X_.astype(target_type)


def decode_data(X, idx2encoder):
    for idx, encoder in idx2encoder.items():
        column = X.columns[idx]
        sub_df = X[[column]]
        sub_df.columns = [0]
        X[[column]] = encoder.inverse_transform(sub_df)  # .astype(target_types[idx])
    return X

"""KNN Imputer for Missing Data"""
# Author: Ashim Bhattarai
# License: GNU General Public License v3 (GPLv3)

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.neighbors._base import _get_weights
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from skimpute.utils import process_dataframe, parse_cat_col, build_encoder, encode_data, decode_data
from skimpute.wrapped_encodes.target_encoder import TargetEncoder
from .pairwise_external import _get_mask
from .pairwise_external import pairwise_distances

__all__ = [
    'KNNImputer',
]
logger = logging.getLogger(__name__)


class KNNImputer(BaseEstimator, TransformerMixin):
    """Imputation for completing missing values using k-Nearest Neighbors.

    Each sample's missing values are imputed using values from ``n_neighbors``
    nearest neighbors found in the training set. Each missing feature is then
    imputed as the average, either weighted or unweighted, of these neighbors.
    Note that if a sample has more than one feature missing, then the
    neighbors for that sample can be different depending on the particular
    feature being imputed. Finally, where the number of donor neighbors is
    less than ``n_neighbors``, the training set average for that feature is
    used during imputation.

    Parameters
    ----------
    missing_values : integer or "NaN", optional (default = "NaN")
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For missing values encoded as
        ``np.nan``, use the string value "NaN".

    n_neighbors : int, optional (default = 5)
        Number of neighboring samples to use for imputation.

    weights : str or callable, optional (default = "uniform")
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    metric : str or callable, optional (default = "masked_euclidean")
        Distance metric for searching neighbors. Possible values:
        - 'masked_euclidean'
        - [callable] : a user-defined function which conforms to the
        definition of _pairwise_callable(X, Y, metric, **kwds). In other
        words, the function accepts two arrays, X and Y, and a
        ``missing_values`` keyword in **kwds and returns a scalar distance
        value.

    row_max_missing : float, optional (default = 0.5)
        The maximum fraction of columns (i.e. features) that can be missing
        before the sample is excluded from nearest neighbor imputation. It
        means that such rows will not be considered a potential donor in
        ``fit()``, and in ``transform()`` their missing feature values will be
        imputed to be the column mean for the entire dataset.

    col_max_missing : float, optional (default = 0.8)
        The maximum fraction of rows (or samples) that can be missing
        for any feature beyond which an error is raised.

    copy : boolean, optional (default = True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, if metric is
        "masked_euclidean" and copy=False then missing_values in the
        input matrix X will be overwritten with zeros.

    Attributes
    ----------
    statistics_ : 1-D array of length {n_features}
        The 1-D array contains the mean of each feature calculated using
        observed (i.e. non-missing) values. This is used for imputing
        missing values in samples that are either excluded from nearest
        neighbors search because they have too many ( > row_max_missing)
        missing features or because all of the sample's k-nearest neighbors
        (i.e., the potential donors) also have the relevant feature value
        missing.

    References
    ----------
    * Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor
      Hastie, Robert Tibshirani, David Botstein and Russ B. Altman, Missing
      value estimation methods for DNA microarrays, BIOINFORMATICS Vol. 17
      no. 6, 2001 Pages 520-525.

    Examples
    --------
    >>> from skimpute import KNNImputer
    >>> nan = float("NaN")
    >>> X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
    >>> imputer = KNNImputer(n_neighbors=2, weights="uniform")
    >>> imputer.fit_transform(X)
    array([[1. , 2. , 4. ],
           [3. , 4. , 3. ],
           [5.5, 6. , 5. ],
           [8. , 8. , 7. ]])
    """

    def __init__(self, missing_values="NaN", n_neighbors=5,
                 weights="uniform", metric="masked_euclidean",
                 row_max_missing=0.5, col_max_missing=0.8, copy=True, consider_ordinal_as_cat=False):

        self.consider_ordinal_as_cat = consider_ordinal_as_cat
        self.missing_values = missing_values
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.row_max_missing = row_max_missing
        self.col_max_missing = col_max_missing
        self.copy = copy

    def _impute(self, dist, X, fitted_X, mask, mask_fx):
        """Helper function to find and impute missing values"""

        # For each column, find and impute
        n_rows_X, n_cols_X = X.shape
        for c in range(n_cols_X):
            if not np.any(mask[:, c], axis=0):
                continue

            # Row index for receivers and potential donors (pdonors)
            receivers_row_idx = np.where(mask[:, c])[0]
            pdonors_row_idx = np.where(~mask_fx[:, c])[0]

            # Impute using column mean if n_neighbors are not available
            if len(pdonors_row_idx) < self.n_neighbors:
                warnings.warn("Insufficient number of neighbors! "
                              "Filling in column mean.")
                X[receivers_row_idx, c] = self.statistics_[c]
                continue

            # Get distance from potential donors
            dist_pdonors = dist[receivers_row_idx][:, pdonors_row_idx]
            dist_pdonors = dist_pdonors.reshape(-1,
                                                len(pdonors_row_idx))

            # Argpartition to separate actual donors from the rest
            pdonors_idx = np.argpartition(
                dist_pdonors, self.n_neighbors - 1, axis=1)

            # Get final donors row index from pdonors
            donors_idx = pdonors_idx[:, :self.n_neighbors]

            # Get weights or None
            dist_pdonors_rows = np.arange(len(donors_idx))[:, None]
            weight_matrix = _get_weights(
                dist_pdonors[
                    dist_pdonors_rows, donors_idx], self.weights)
            donor_row_idx_ravel = donors_idx.ravel()

            # Retrieve donor values and calculate kNN score
            fitted_X_temp = fitted_X[pdonors_row_idx]
            donors = fitted_X_temp[donor_row_idx_ravel, c].reshape(
                (-1, self.n_neighbors))
            donors_mask = _get_mask(donors, self.missing_values)
            donors = np.ma.array(donors, mask=donors_mask)

            # Final imputation
            imputed = np.ma.average(donors, axis=1,
                                    weights=weight_matrix)
            X[receivers_row_idx, c] = imputed.data
        return X

    def fit(self, X, y=None):
        """Fit the imputer on X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns self.
        """
        self.dtypes = X.dtypes

        X_ = process_dataframe(X)
        self.num_idx, self.cat_idx = parse_cat_col(X_, self.consider_ordinal_as_cat)
        encoder = TargetEncoder()
        self.idx2encoder, X_, _ = build_encoder(X_, y, self.cat_idx, encoder, [], "float32")
        # Check if % missing in any column > col_max_missing
        mask = _get_mask(X_, self.missing_values)
        self.do_simple_imputing = False
        if np.any(mask.sum(axis=0) > (X_.shape[0] * self.col_max_missing)):
            self.do_simple_imputing = True
            logger.warning("Some column(s) have more than {}% missing values".format(self.col_max_missing * 100))
        X_col_means = np.ma.array(X_, mask=mask).mean(axis=0).data

        # Check if % missing in any row > row_max_missing
        bad_rows = mask.sum(axis=1) > (mask.shape[1] * self.row_max_missing)
        if np.any(bad_rows):
            logger.warning(
                "There are rows with more than {0}% missing values. These "
                "rows are not included as donor neighbors."
                    .format(self.row_max_missing * 100))

            # Remove rows that have more than row_max_missing % missing
            X_ = X_.iloc[~bad_rows, :]

        # Check if sufficient neighboring samples available
        if X_.shape[0] < self.n_neighbors:
            self.do_simple_imputing = True
            logger.warning("There are only %d samples, but n_neighbors=%d." % (X_.shape[0], self.n_neighbors))
        self.fitted_X_ = X_.values
        self.statistics_ = X_col_means
        return self

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            The input data to complete.

        Returns
        -------
        X : {array-like}, shape = [n_samples, n_features]
            The imputed dataset.
        """

        check_is_fitted(self, ["fitted_X_", "statistics_", "do_simple_imputing", "idx2encoder", "dtypes"])
        if self.do_simple_imputing:
            logger.debug("KNNImputer is doing adaptive simple imputation.")
            from skimpute import AdaptiveSimpleImputer
            return AdaptiveSimpleImputer(consider_ordinal_as_cat=self.consider_ordinal_as_cat).fit_transform(X)
        X_ = process_dataframe(X)
        columns = X.columns
        index = X.index
        X_ = encode_data(X_, self.idx2encoder, "float32")
        # Get fitted data and ensure correct dimension
        n_rows_fit_X, n_cols_fit_X = self.fitted_X_.shape
        n_rows_X, n_cols_X = X_.shape

        if n_cols_X != n_cols_fit_X:
            raise ValueError("Incompatible dimension between the fitted "
                             "dataset and the one to be transformed.")
        mask = _get_mask(X_, self.missing_values)

        row_total_missing = mask.sum(axis=1)
        if not np.any(row_total_missing):
            return X_

        # Check for excessive missingness in rows
        bad_rows = row_total_missing > (mask.shape[1] * self.row_max_missing)
        if np.any(bad_rows):
            logger.warning(
                "There are rows with more than {0}% missing values. The "
                "missing features in these rows are imputed with column means."
                    .format(self.row_max_missing * 100))
            X_bad = X_.iloc[bad_rows, :]
            X_ = X_.iloc[~bad_rows, :]
            mask = mask[~bad_rows]
            row_total_missing = mask.sum(axis=1)
        row_has_missing = row_total_missing.astype(np.bool)

        if np.any(row_has_missing):
            # Mask for fitted_X
            mask_fx = _get_mask(self.fitted_X_, self.missing_values)

            # Pairwise distances between receivers and fitted samples
            dist = np.empty((len(X_), len(self.fitted_X_)))
            X_miss = X_.iloc[row_has_missing].values
            fitted_X = self.fitted_X_

            # X_miss, self.scaler = self.min_max_scale(X_miss)
            # fitted_X, _ = self.min_max_scale(fitted_X)

            dist[row_has_missing] = pairwise_distances(
                X_miss, fitted_X, metric=self.metric,
                squared=False, missing_values=self.missing_values)

            # Find and impute missing
            X_ = self._impute(dist, X_.values, self.fitted_X_, mask, mask_fx)

        # Merge bad rows to X and mean impute their missing values
        if np.any(bad_rows):
            bad_missing_index = np.where(_get_mask(X_bad, self.missing_values))
            X_bad[bad_missing_index] = np.take(self.statistics_,
                                               bad_missing_index[1])
            X_merged = np.empty((n_rows_X, n_cols_X))
            X_merged[bad_rows, :] = X_bad
            X_merged[~bad_rows, :] = X_
            X_ = X_merged
        X = pd.DataFrame(X_, columns=columns, index=index)
        X = decode_data(X, self.idx2encoder)
        X = X.astype(self.dtypes)
        return X

    def min_max_scale(self, X: np.ndarray):
        X_miss_mask = _get_mask(X, self.missing_values)
        X = SimpleImputer(strategy="most_frequent").fit_transform(X)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        X[X_miss_mask] = np.nan
        return X, scaler

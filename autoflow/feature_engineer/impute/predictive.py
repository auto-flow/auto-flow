#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import functools
from collections import Counter
from copy import deepcopy
from time import time

import numpy as np
import pandas as pd
from frozendict import frozendict
from sklearn import clone
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from .base import BaseImputer
from .simple import CategoricalImputer, SimpleImputer
from .utils import build_encoder, decode_data


class PredictiveImputer(BaseImputer):
    reg_cls = None
    clf_cls = None

    def __init__(
            self,
            categorical_feature=None,
            numerical_feature=None,
            copy=False,
            missing_rate=0.4,
            max_iter=10,
            decreasing=False,
            budget=10,
            verbose=0,
            params=frozendict()
    ):
        super(PredictiveImputer, self).__init__(
            categorical_feature=categorical_feature,
            numerical_feature=numerical_feature,
            copy=copy,
            missing_rate=missing_rate
        )
        self.params = params
        self.budget = budget
        self.decreasing = decreasing
        self.max_iter = max_iter
        self.verbose = verbose
        self.logging_level = 20 if verbose > 0 else 20

    def _predictive_impute(self, Ximp, mask):
        """The missForest algorithm"""
        self.cat_idx = np.arange(Ximp.shape[1])[Ximp.columns.isin(self.categorical_feature)]
        self.num_idx = np.arange(Ximp.shape[1])[Ximp.columns.isin(self.numerical_feature)]
        # Count missing per column
        if isinstance(Ximp, pd.DataFrame):
            Ximp = Ximp.values
        nan_cat_cols = np.count_nonzero(np.sum(mask[:, self.cat_idx], axis=0))
        nan_num_cols = np.count_nonzero(np.sum(mask[:, self.num_idx], axis=0))
        self.logger.log(self.logging_level,
                        f"X contains {nan_cat_cols} categorical-missing columns, {nan_num_cols} numerical-missing columns.")
        col_missing_count = mask.sum(axis=0)

        # Get col and row indices for missing
        missing_rows, missing_cols = np.where(mask)
        regressor = classifier = n_catmissing = None
        if self.num_idx.size:
            # Only keep indices for numerical vars
            keep_idx_num = np.in1d(missing_cols, self.num_idx)
            missing_num_rows = missing_rows[keep_idx_num]
            missing_num_cols = missing_cols[keep_idx_num]
            # Make initial guess for missing values
            col_means = np.full(Ximp.shape[1], fill_value=np.nan)
            col_means[self.num_idx] = deepcopy(self.statistics_.get('col_means'))
            Ximp[missing_num_rows, missing_num_cols] = np.take(
                col_means, missing_num_cols)
            # Instantiate regression model
            regressor = self.reg_cls(**self.update_params(self.params, "regression"))
        # If needed, repeat for categorical variables
        if self.cat_idx.size:
            # Calculate total number of missing categorical values (used later)
            n_catmissing = np.sum(mask[:, self.cat_idx])
            self.logger.log(self.logging_level, f"n_catmissing = {n_catmissing}")
            # Only keep indices for categorical vars
            keep_idx_cat = np.in1d(missing_cols, self.cat_idx)
            missing_cat_rows = missing_rows[keep_idx_cat]
            missing_cat_cols = missing_cols[keep_idx_cat]
            # Make initial guess for missing values
            col_modes = np.full([Ximp.shape[1]], np.nan)
            col_modes[self.cat_idx] = self.encoded_col_modes.copy()
            Ximp[missing_cat_rows, missing_cat_cols] = np.take(col_modes, missing_cat_cols)
            # Instantiate classification model
            classifier = self.clf_cls(**self.update_params(self.params, "classification"))

        # 2. misscount_idx: sorted indices of cols in X based on missing count
        misscount_idx = np.argsort(col_missing_count)
        # Reverse order if decreasing is set to True
        if self.decreasing is True:
            misscount_idx = misscount_idx[::-1]

        # 3. While new_gammas < old_gammas & self.iter_count_ < max_iter loop:
        self.iter = 0
        self.gamma_history = []
        self.gamma_cat_history = []
        self.cost_times = []
        gamma_new = 0
        gamma_old = np.inf
        gamma_newcat = 0
        gamma_oldcat = np.inf
        col_index = np.arange(Ximp.shape[1])
        start_time = time()
        cost_time = 0
        self.logger.log(self.logging_level, "-" * 50)
        while (
                gamma_new < gamma_old or gamma_newcat < gamma_oldcat) and \
                self.iter < self.max_iter:

            # 4. store previously imputed matrix
            Ximp_old = Ximp.copy()
            if self.iter != 0:
                gamma_old = gamma_new
                gamma_oldcat = gamma_newcat
            # 5. loop
            for s in misscount_idx:
                # Column indices other than the one being imputed
                s_prime = np.delete(col_index, s)

                # Get indices of rows where 's' is observed and missing
                obs_rows = np.where(~mask[:, s])[0]
                mis_rows = np.where(mask[:, s])[0]

                # If no missing, then skip
                if len(mis_rows) == 0:
                    continue

                # Get observed values of 's'
                yobs = Ximp[obs_rows, s]
                n_yobs = np.unique(yobs).size
                # Get 'X' for both observed and missing 's' column
                xobs = Ximp[np.ix_(obs_rows, s_prime)]
                xmis = Ximp[np.ix_(mis_rows, s_prime)]

                # 6. Fit a random forest over observed and predict the missing
                if self.cat_idx.size and s in self.cat_idx:
                    yobs = yobs.astype('int32')
                    classifier = clone(classifier)
                    if n_yobs > 1:
                        classifier.fit(X=xobs, y=yobs)
                        # 7. predict ymis(s) using xmis(x)
                        ymis = classifier.predict(xmis)
                    else:
                        ymis = np.zeros_like(mis_rows) - 1
                        self.logger.warning(f"in column index {s}, all value are same,"
                                            f" var = 0, don not use classifier to fit .")
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis
                else:
                    yobs = yobs.astype('float32')
                    regressor = clone(regressor)
                    regressor.fit(X=xobs, y=yobs)
                    # 7. predict ymis(s) using xmis(x)
                    ymis = regressor.predict(xmis)
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis

            # 9. Update gamma (stopping criterion)
            if self.cat_idx.size:
                gamma_newcat = np.sum(
                    (Ximp[:, self.cat_idx] != Ximp_old[:, self.cat_idx])) / n_catmissing
            if self.num_idx.size:
                gamma_new = np.sum((Ximp[:, self.num_idx] - Ximp_old[:, self.num_idx]) ** 2) / np.sum(
                    (Ximp[:, self.num_idx]) ** 2)

            self.logger.log(self.logging_level,
                            f"{self.iter} | gamma_new = {gamma_new:.3f}, gamma_old = {gamma_old:.3f}")
            self.logger.log(self.logging_level,
                            f"{self.iter} | gamma_newcat = {gamma_newcat:.3f}, gamma_oldcat = {gamma_oldcat:.3f}")
            self.logger.log(self.logging_level,
                            f"{self.iter} | {self.__class__.__name__} Coverage Iteration: {self.iter}")
            self.logger.log(self.logging_level, "-" * 50)

            cost_time = time() - start_time
            gamma_new=float(gamma_new)
            if np.isinf(gamma_new):
                gamma_new=0
            gamma_newcat=float(gamma_newcat)
            if np.isinf(gamma_newcat):
                gamma_newcat=0
            self.gamma_history.append(gamma_new)
            self.gamma_cat_history.append(gamma_newcat)
            self.cost_times.append(cost_time)
            if cost_time > self.budget:
                self.logger.log(self.logging_level,
                                f"cost_time = {cost_time:.2f}s, {self.__class__.__name__} early stopping ... ")
                break
            self.iter += 1
        self.logger.log(self.logging_level,
                        f"{self.__class__.__name__}'s cost_time = {cost_time:.2f}s , budget = {self.budget}s ")
        return Ximp

    def update_params(self, params, problem_type):
        raise NotImplementedError

    def fit(self, X, y=None, categorical_feature=None, numerical_feature=None, **kwargs):
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

        X = super(PredictiveImputer, self).fit(X, y, categorical_feature, numerical_feature)
        # Check if any column has all missing
        mask = pd.isna(X).values
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")

        # Now, make initial guess for missing values
        col_means = np.nanmean(X[self.numerical_feature], axis=0) if self.numerical_feature.size else None
        col_modes = CategoricalImputer(strategy="most_frequent") \
            .fit(X[self.categorical_feature]).statistics_ if self.categorical_feature.size else None

        self.statistics_ = {"col_means": col_means, "col_modes": col_modes}
        if col_means is not None:
            self.logger.log(self.logging_level, "col_means statistics:")
            for num_col, mean in zip(self.numerical_feature, col_means):
                self.logger.log(self.logging_level, f"{num_col}\t:\t{mean:.2f}")
        if col_modes is not None:
            self.logger.log(self.logging_level, "col_modes statistics:")
            for cat_col, mode in zip(self.categorical_feature, col_modes):
                self.logger.log(self.logging_level,
                                f"{cat_col}\t:\t{mode}\t"
                                f"({Counter(X[cat_col]).most_common()[0][1]} times)")
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
        # Confirm whether fit() has been called
        check_is_fitted(self, ["statistics_"])
        X = super(PredictiveImputer, self).transform(X)
        dtypes = X.dtypes
        # Check if any column has all missing
        mask = pd.isna(X).values
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            self.logger.warning("One or more columns have all rows missing. Using BaseImputer to do imputing.")
            return SimpleImputer(
                categorical_feature=self.categorical_feature,
                numerical_feature=self.numerical_feature,
                copy=self.copy,
            ).fit_transform(X)

        if not mask.sum() > 0:
            self.logger.warning("No missing value located; returning original "
                                "dataset.")
            return X

        # convert string column to index
        col_modes = self.statistics_["col_modes"]
        column2coder, X_, additional_data_list = build_encoder(
            X, None, self.categorical_feature,
            OrdinalEncoder(), [col_modes],
            "float32", functools.partial(self.logger.log, self.logging_level))
        self.encoded_col_modes = additional_data_list[0].astype('float32')
        # Call missForest function to impute missing
        columns = X_.columns
        index = X_.index
        Ximp = self._predictive_impute(X_, mask)
        X = pd.DataFrame(Ximp, columns=columns, index=index)
        X = decode_data(X, column2coder)
        X = X.astype(dtypes)
        # Return imputed dataset
        return X

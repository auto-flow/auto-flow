# -*- coding: utf-8 -*-
# Author: Franziska Horn <cod3licious@gmail.com>
# License: MIT

from __future__ import unicode_literals, division, print_function, absolute_import

import warnings
from builtins import zip
from collections import Counter
from time import time

import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            problem_type="regression",
            featsel_runs=5,
            keep=None,
            n_jobs=1,
            verbose=0,
            random_state=42,
            consider_other=False,
            regularization="l1"
    ):
        """
        multi-step cross-validated feature selection

        Inputs:
            - problem_type: str, either "regression" or "classification" (default: "regression")
            - featsel_runs: number of times to perform in the feature selection part with a random fraction of data points (int; default: 5)
            - keep: list of features that should be kept no matter what
            - n_jobs: how many jobs to run when selecting the features in parallel (int; default: 1)
            - verbose: verbosity level (int; default: 0)

        Attributes:
            - good_cols_: list of good features (to select via pandas DataFrame columns)
            - original_columns_: original columns of X when calling fit
            - return_df_: whether fit was called with a dataframe in which case a df will be returned as well,
                          otherwise a numpy array
        """
        self.regularization = regularization
        self.consider_other = consider_other
        self.problem_type = problem_type
        self.featsel_runs = featsel_runs
        self.keep = keep
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.rng = check_random_state(random_state)

    def _add_noise_features(self, X):
        """
        Adds 3-1.5*d additional noise features to X.

        Inputs:
            - X: n x d numpy array with d features
        Returns:
            - X with additional noise features
        """
        n_feat = X.shape[1]
        if X.shape[0] > 50 and n_feat > 1:
            # shuffled features
            rand_noise = StandardScaler().fit_transform(self.rng.permutation(X.flatten()).reshape(X.shape))
            X = np.hstack([X, rand_noise])
        # normally distributed noise
        rand_noise = self.rng.randn(X.shape[0], max(3, int(0.5 * n_feat)))
        X = np.hstack([X, rand_noise])
        return X

    def _noise_filtering(self, X, target, good_cols=()):
        """
        Trains a prediction model with additional noise features and selects only those of the
        original features that have a higher coefficient than any of the noise features.

        Inputs:
            - X: n x d numpy array with d features
            - target: n dimensional array with targets corresponding to the data points in X
            - good_cols: list of column names for the features in X
            - problem_type: str, either "regression" or "classification" (default: "regression")
        Returns:
            - good_cols: list of noise filtered column names
        """
        problem_type = self.problem_type
        good_cols = list(good_cols)
        n_feat = X.shape[1]
        assert len(good_cols) == n_feat, "fewer column names provided than features in X."
        if not good_cols:
            good_cols = list(range(n_feat))
        # perform noise filtering on these features
        model = self.get_model()
        if model is not None:
            X = self._add_noise_features(X)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # TODO: remove if sklearn least_angle issue is fixed
                try:
                    model = self.iterative_training(model, X, target)
                except ValueError:
                    rand_idx = self.rng.permutation(X.shape[0])
                    model = self.iterative_training(model, X[rand_idx], target[rand_idx])
            if problem_type == "regression":
                coefs = np.abs(model.coef_)
            else:
                # model.coefs_ is n_classes x n_features, but we need n_features
                coefs = np.max(np.abs(model.coef_), axis=0)
            weights = dict(zip(good_cols, coefs[:len(good_cols)]))
            # only include features that are more important than our known noise features
            noise_w_thr = np.max(coefs[n_feat:])
            good_cols = [c for c in good_cols if weights[c] > noise_w_thr]
        return good_cols

    def get_model(self):
        if self.problem_type == "regression":
            if self.regularization == "l1":
                model = lm.LassoLars(eps=1e-8)
            elif self.regularization == "l2":
                model = lm.Ridge(normalize=True, random_state=self.random_state)
            else:
                raise ValueError(f"Unknown regularization {self.regularization}")
        elif self.problem_type == "classification":
            if self.regularization == "l1":
                model = lm.LogisticRegression(penalty="l1", solver="saga", class_weight="balanced",
                                              random_state=self.random_state, warm_start=True)
            elif self.regularization == "l2":
                model = lm.LogisticRegression(penalty="l2", class_weight="balanced",
                                              random_state=self.random_state, warm_start=True)
            else:
                raise ValueError(f"Unknown regularization {self.regularization}")
        else:
            raise ValueError("Unknown problem_type %r - not performing noise filtering." % self.problem_type)
        return model

    def iterative_training(self, model, X, y):
        start_time = time()
        if getattr(model, "warm_start", False) == False:
            return model.fit(X, y)
        for max_iter in range(10, 110, 10):
            model.max_iter = max_iter
            model.fit(X, y)
            if time() - start_time > 10:
                break
        return model

    def _select_features_1run(self, df, target):
        """
        One feature selection run.

        Inputs:
            - df: nxp pandas DataFrame with n data points and p features; to avoid overfitting, only provide data belonging
                  to the n training data points. The variables have to be scaled to have 0 mean and unit variance.
            - target: n dimensional array with targets corresponding to the data points in df
            - problem_type: str, either "regression" or "classification" (default: "regression")
            - verbose: verbosity level (int; default: 0)
        Returns:
            - good_cols: list of column names for df with which a prediction model can be trained
        """
        # initial selection of too few but (hopefully) relevant features
        problem_type = self.problem_type
        verbose = self.verbose
        consider_other = self.consider_other
        model = self.get_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # TODO: remove if sklearn least_angle issue is fixed
            try:
                model = self.iterative_training(model, df, target)
            except ValueError:
                # try once more with shuffled data, if it still doesn't work, give up
                rand_idx = self.rng.permutation(df.shape[0])
                model = self.iterative_training(model, df.iloc[rand_idx], target[rand_idx])
        if problem_type == "regression":
            coefs = np.abs(model.coef_)
        else:
            # model.coefs_ is n_classes x n_features, but we need n_features
            coefs = np.max(np.abs(model.coef_), axis=0)
        # weight threshold: select at most 0.2*n_train initial features
        thr = sorted(coefs, reverse=True)[min(df.shape[1] - 1, df.shape[0] // 5)]
        initial_cols = list(df.columns[coefs > thr])
        # noise filter
        initial_cols = self._noise_filtering(df[initial_cols].to_numpy(), target, initial_cols)
        good_cols = set(initial_cols)
        if verbose > 0:
            print("[featsel]\t %i initial features." % len(initial_cols))
        if not consider_other:
            return sorted(list(good_cols))  # fixme set maybe randomly
        # add noise features
        X_w_noise = self._add_noise_features(df[initial_cols].to_numpy())
        # go through all remaining features in splits of n_feat <= 0.5*n_train
        other_cols = list(self.rng.permutation(list(set(df.columns).difference(initial_cols))))
        if other_cols:
            n_splits = int(np.ceil(len(other_cols) / max(10, 0.5 * df.shape[0] - len(initial_cols))))
            split_size = int(np.ceil(len(other_cols) / n_splits))
            for i in range(n_splits):
                current_cols = other_cols[i * split_size:min(len(other_cols), (i + 1) * split_size)]
                X = np.hstack([df[current_cols].to_numpy(), X_w_noise])
                model = self.get_model()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # TODO: remove if sklearn least_angle issue is fixed
                    try:
                        model = self.iterative_training(model, X, target)
                    except ValueError:
                        rand_idx = self.rng.permutation(X.shape[0])
                        model = self.iterative_training(model, X[rand_idx], target[rand_idx])
                current_cols.extend(initial_cols)
                if problem_type == "regression":
                    coefs = np.abs(model.coef_)
                else:
                    # model.coefs_ is n_classes x n_features, but we need n_features
                    coefs = np.max(np.abs(model.coef_), axis=0)
                weights = dict(zip(current_cols, coefs[:len(current_cols)]))
                # only include features that are more important than our known noise features
                noise_w_thr = np.max(coefs[len(current_cols):])
                good_cols.update([c for c in weights if abs(weights[c]) > noise_w_thr])
                if verbose > 0:
                    print(
                        "[featsel]\t Split %2i/%i: %3i candidate features identified." % (
                            i + 1, n_splits, len(good_cols)),
                        end="\r")
        # noise filtering on the combination of features
        good_cols = list(good_cols)
        good_cols = self._noise_filtering(df[good_cols].to_numpy(), target, good_cols)
        if verbose > 0:
            print("\n[featsel]\t Selected %3i features after noise filtering." % len(good_cols))
        return good_cols

    def select_features(self, df, target):
        """
        Selects predictive features given the data and targets.

        Inputs:
            - df: nxp pandas DataFrame with n data points and p features; to avoid overfitting, only provide data belonging
                  to the n training data points.
            - target: n dimensional array with targets corresponding to the data points in df
            - featsel_runs: number of times to perform in the feature selection part with a random fraction of data points (int; default: 5)
            - keep: list of features that should be kept no matter what
            - problem_type: str, either "regression" or "classification" (default: "regression")
            - n_jobs: how many jobs to run when selecting the features in parallel (int; default: 1)
            - verbose: verbosity level (int; default: 0)
        Returns:
            - good_cols: list of column names for df with which a regression model can be trained
        """
        featsel_runs = self.featsel_runs
        keep = self.keep
        problem_type = self.problem_type
        n_jobs = self.n_jobs
        verbose = self.verbose
        consider_other = self.consider_other

        if not (len(df) == len(target)):
            raise ValueError("[featsel] df and target dimension mismatch.")
        if keep is None:
            keep = []
        # check that keep columns are actually in df (- the columns might have been transformed to strings!)
        keep = [c for c in keep if c in df.columns and not str(c) in df.columns] + [str(c) for c in keep if
                                                                                    str(c) in df.columns]
        # scale features to have 0 mean and unit std
        if verbose > 0:
            if featsel_runs > df.shape[0]:
                print("[featsel] WARNING: Less data points than featsel runs!!")
            print("[featsel] Scaling data...", end="")
        scaler = StandardScaler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, dtype=np.float32)
            if problem_type == "regression":
                target_scaled = scaler.fit_transform(target.reshape(-1, 1)).ravel()
            else:
                target_scaled = target
        if verbose > 0:
            print("done.")

        good_cols = list(df.columns)

        # select good features in k runs in parallel
        # by doing sort of a cross-validation (i.e., randomly subsample data points)
        def run_select_features(i):
            if verbose > 0:
                print("[featsel] Feature selection run %i/%i" % (i + 1, featsel_runs))
            rand_idx = self.rng.permutation(df_scaled.index)[:max(10, int(0.85 * len(df_scaled)))]
            return self._select_features_1run(df_scaled.iloc[rand_idx], target_scaled[rand_idx])

        if featsel_runs >= 1 and problem_type in ("regression", "classification"):
            if n_jobs == 1 or featsel_runs == 1:
                # only use parallelization code if you actually parallelize
                selected_columns = []
                for i in range(featsel_runs):
                    selected_columns.extend(run_select_features(i))
            else:
                def flatten_lists(l):
                    return [item for sublist in l for item in sublist]

                n_jobs = min(featsel_runs, n_jobs)
                selected_columns = flatten_lists(Parallel(n_jobs=n_jobs, verbose=100 * verbose)(
                    delayed(run_select_features)(i) for i in range(featsel_runs)))

            if selected_columns:
                selected_columns = Counter(selected_columns)
                # sort by frequency, but down weight longer formulas to break ties
                selected_columns = sorted(selected_columns, key=lambda x: selected_columns[x] - 0.000001 * len(str(x)),
                                          reverse=True)
                if verbose > 0:
                    print(
                        "[featsel] %i features after %i feature selection runs" % (len(selected_columns), featsel_runs))
                # correlation filtering
                selected_columns = keep + [c for c in selected_columns if c not in keep]
                if not keep:
                    good_cols = [selected_columns[0]]
                    k = 1
                else:
                    good_cols = keep
                    k = len(keep)
                if len(selected_columns) > k:
                    correlations = df_scaled[selected_columns].corr()
                    for i, c in enumerate(selected_columns[k:], k):
                        # only take features that are somewhat uncorrelated with the rest
                        if np.max(np.abs(correlations[c].ravel()[:i])) < 0.9:
                            good_cols.append(c)
                if verbose > 0:
                    print("[featsel] %i features after correlation filtering" % len(good_cols))

        # perform noise filtering on these features
        good_cols = self._noise_filtering(df_scaled[good_cols].to_numpy(), target_scaled, good_cols)
        if verbose > 0:
            print("[featsel] %i features after noise filtering" % len(good_cols))
            if not good_cols:
                print("[featsel] WARNING: Not a single good features was found...")

        # add keep columns back in
        good_cols = keep + [c for c in good_cols if c not in keep]
        if verbose > 0 and keep:
            print(
                "[featsel] %i final features selected (including %i original keep features)." % (
                    len(good_cols), len(keep)))
        return good_cols

    def fit(self, X, y):
        """
        Selects features.

        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
            - y: pandas dataframe or numpy array with targets for all n_datapoints
        """
        self.return_df_ = isinstance(X, pd.DataFrame)
        # store column names as they'll be lost in the other check
        # first calling np.array assures that all the column names have the same dtype
        # as otherwise we get problems when calling np.random.permutation on the columns
        cols = list(np.array(list(X.columns))) if isinstance(X, pd.DataFrame) else []
        # check input variables
        X, target = check_X_y(X, y, y_numeric=self.problem_type == "regression")
        if not cols:
            cols = ["x%i" % i for i in range(X.shape[1])]
        self.original_columns_ = cols
        # transform X into a dataframe (again)
        df = pd.DataFrame(X, columns=cols)
        # do the feature selection
        self.good_cols_ = self.select_features(df, target)
        return self

    def transform(self, X):
        """
        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
        Returns:
            - new_X: new pandas dataframe or numpy array with only the good features
        """
        check_is_fitted(self, ["good_cols_"])
        if not self.good_cols_:
            if self.verbose > 0:
                print("[FeatureSelector] WARNING: No good features found; returning data unchanged.")
            return X
        # store column names as they'll be lost in the other check
        # first calling np.array assures that all the column names have the same dtype
        # as otherwise we get problems when calling np.random.permutation on the columns
        cols = list(np.array(list(X.columns))) if isinstance(X, pd.DataFrame) else []
        # check input variables
        X = check_array(X, force_all_finite="allow-nan")
        if not cols:
            cols = ["x%i" % i for i in range(X.shape[1])]
        if not cols == self.original_columns_:
            raise ValueError("[FeatureSelector] Not the same features as when calling fit.")
        # transform X into a dataframe (again) and select columns
        new_X = pd.DataFrame(X, columns=cols)[self.good_cols_]
        # possibly transform into a numpy array
        if not self.return_df_:
            new_X = new_X.to_numpy()
        return new_X

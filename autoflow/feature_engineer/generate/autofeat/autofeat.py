# -*- coding: utf-8 -*-
# Author: Franziska Horn <cod3licious@gmail.com>
# License: MIT

from __future__ import unicode_literals, division, print_function, absolute_import

from builtins import range
from copy import copy
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sympy.utilities.lambdify import lambdify

from autoflow.constants import VARIABLE_PATTERN
from autoflow.feature_engineer.select import BorutaFeatureSelector
from autoflow.utils.data import check_n_jobs
from autoflow.utils.logging_ import get_logger
from .feateng import engineer_features, n_cols_generated, colnames2symbols
from .featsel import FeatureSelector


class AutoFeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            problem_type=None,
            categorical_cols=None,
            feateng_cols=None,
            units=None,
            max_used_feats=10,
            feateng_steps=2,
            featsel_runs=3,
            max_gb=None,
            transformations=None,
            apply_pi_theorem=True,
            always_return_numpy=False,
            n_jobs=-1,
            verbose=0,
            random_state=0,
            consider_other=False,
            regularization=None,
            div_op=True,
            exp_op=False,
            log_op=False,
            abs_op=False,
            sqrt_op=False,
            sqr_op=True,
            do_final_selection=False,
            standardize=False
    ):
        """
        multi-step feature engineering and cross-validated feature selection to generate promising additional
        features for your dataset and train a linear prediction model with them.

        Inputs:
            - problem_type: str, either "regression" or "classification" (default: "regression")
            - categorical_cols: list of column names of categorical features; these will be transformed into
                                0/1 encoding (default: None)
            - feateng_cols: list of column names that should be used for the feature engineering part
                            (default None --> all, with categorical_cols in 0/1 encoding)
            - feateng_steps: number of steps to perform in the feature engineering part (int; default: 2)
            - featsel_runs: number of times to perform in the feature selection part with a random fraction of data points (int; default: 5)
            - max_gb: if an int is given: maximum number of gigabytes to use in the process (i.e. mostly the
                      feature engineering part). this is no guarantee! it will lead to subsampling of the
                      data points if the new dataframe generated is n_rows * n_cols * 32bit > max_gb
                      Note: this is only an approximate estimate of the final matrix; intermediate representations could easily
                            take up at least 2 or 3 times that much space...If you can, subsample before, you know your data best.
            - transformations: list of transformations that should be applied; possible elements:
                               "1/", "exp", "log", "abs", "sqrt", "^2", "^3", "1+", "1-", "sin", "cos", "exp-", "2^"
                               (first 7, i.e., up to ^3, are applied by default)
            - apply_pi_theorem: whether or not to apply the pi theorem (if units are given; bool; default: True)
            - always_return_numpy: whether to always return a numpy array instead of a pd dataframe when calling (fit_)transform
                                   (default: False; mainly used for sklearn estimator checks)
            - n_jobs: how many jobs to run when selecting the features in parallel (int; default: 1)
            - verbose: verbosity level (int; default: 0)

        Attributes:
            - original_columns_: original columns of X when calling fit
            - all_columns_: columns of X after calling fit
            - categorical_cols_map_: dict mapping from the original categorical columns to a list with new column names
            - feateng_cols_: actual columns used for the feature engineering
            - feature_formulas_: sympy formulas to generate new features
            - feature_functions_: compiled feature functions with columns
            - new_feat_cols_: list of good new features that should be generated when calling transform()
            - good_cols_: columns selected in the feature selection process, used with the final prediction model
            - prediction_model_: sklearn model instance used for the predictions

        Note: when giving categorical_cols or feateng_cols, X later (i.e. when calling fit/fit_transform) has to be a DataFrame
        """
        self.logger = get_logger(self)
        self.standardize = standardize
        self.do_final_selection = do_final_selection
        if transformations is None:
            transformations = []
            if div_op:
                transformations.append("1/")
            if exp_op:
                transformations.append("exp")
            if log_op:
                transformations.append("log")
            if abs_op:
                transformations.append("abs")
            if sqrt_op:
                transformations.append("sqrt")
            if sqr_op:
                transformations.append("^2")
        self.sqr_op = sqr_op
        self.sqrt_op = sqrt_op
        self.abs_op = abs_op
        self.log_op = log_op
        self.exp_op = exp_op
        self.div_op = div_op
        self.regularization = regularization
        self.consider_other = consider_other
        self.random_state = random_state
        self.max_used_feats = max_used_feats
        self.problem_type = problem_type
        self.categorical_cols = categorical_cols
        self.feateng_cols = feateng_cols
        self.units = units
        self.feateng_steps = feateng_steps
        self.max_gb = max_gb
        self.featsel_runs = featsel_runs
        self.transformations = transformations
        self.apply_pi_theorem = apply_pi_theorem
        self.always_return_numpy = always_return_numpy
        self.n_jobs = check_n_jobs(n_jobs)
        self.verbose = verbose

    def __getstate__(self):
        """
        get dict for pickling without feature_functions as they are not pickleable
        """
        return {k: self.__dict__[k] if k != "feature_functions_" else {} for k in self.__dict__}

    def _transform_categorical_cols(self, df):
        """
        Transform categorical features into 0/1 encoding.

        Inputs:
            - df: pandas dataframe with original features
        Returns:
            - df: dataframe with categorical features transformed into multiple 0/1 columns
        """
        self.categorical_cols_map_ = {}
        if self.categorical_cols:
            e = OneHotEncoder(sparse=False, categories="auto")
            for c in self.categorical_cols:
                if c not in df.columns:
                    raise ValueError("[AutoFeat] categorical_col %r not in df.columns" % c)
                ohe = e.fit_transform(df[c].to_numpy()[:, None])
                new_cat_cols = ["cat_%s_%r" % (str(c), i) for i in e.categories_[0]]
                self.categorical_cols_map_[c] = new_cat_cols
                df = df.join(pd.DataFrame(ohe, columns=new_cat_cols, index=df.index))
            # remove the categorical column from our columns to consider
            df.drop(columns=self.categorical_cols, inplace=True)
        return df

    def _generate_features(self, df, new_feat_cols):
        """
        Generate additional features based on the feature formulas for all data points in the df.
        Only works after the model was fitted.

        Inputs:
            - df: pandas dataframe with original features
            - new_feat_cols: names of new features that should be generated (keys of self.feature_formulas_)
        Returns:
            - df: dataframe with the additional feature columns added
        """
        check_is_fitted(self, ["feature_formulas_"])
        if not new_feat_cols:
            return df
        if not new_feat_cols[0] in self.feature_formulas_:
            raise RuntimeError("[AutoFeat] First call fit or fit_transform to generate the features!")
        if self.verbose:
            print("[AutoFeat] Computing %i new features." % len(new_feat_cols))
        # generate all good feature; unscaled this time
        feat_array = np.zeros((len(df), len(new_feat_cols)))
        for i, expr in enumerate(new_feat_cols):
            if self.verbose:
                print("[AutoFeat] %5i/%5i new features" % (i, len(new_feat_cols)), end="\r")
            if expr not in self.feature_functions_:
                # generate a substitution expression based on all the original symbols of the original features
                # for the given generated feature in good cols
                # since sympy can handle only up to 32 original features in ufunctify, we need to check which features
                # to consider here, therefore perform some crude check to limit the number of features used
                cols = [c for i, c in enumerate(self.feateng_cols_) if colnames2symbols(c, i) in expr]
                if not cols:
                    # this can happen if no features were selected and the expr is "E" (i.e. the constant e)
                    f = None
                else:
                    try:
                        f = lambdify([self.feature_formulas_[c] for c in cols], self.feature_formulas_[expr])
                    except Exception:
                        print("[AutoFeat] Error while processing expression: %r" % expr)
                        raise
                self.feature_functions_[expr] = (cols, f)
            else:
                cols, f = self.feature_functions_[expr]
            if f is not None:
                # only generate features for completely not-nan rows
                not_na_idx = df[cols].notna().all(axis=1)
                try:
                    feat_array[not_na_idx, i] = f(*(df[c].to_numpy(dtype=float)[not_na_idx] for c in cols))
                    feat_array[~not_na_idx, i] = np.nan
                except RuntimeWarning:
                    print("[AutoFeat] WARNING: Problem while evaluating expression: %r with columns %r" % (expr, cols),
                          " - is the data in a different range then when calling .fit()? Are maybe some values 0 that shouldn't be?")
                    raise
        if self.verbose:
            print("[AutoFeat] %5i/%5i new features ...done." % (len(new_feat_cols), len(new_feat_cols)))
        df = df.join(pd.DataFrame(feat_array, columns=new_feat_cols, index=df.index))
        return df

    def convert_colname_to_variables(self, df):
        ix = 0
        origin_columns = []
        new_columns = []
        keep_columns = []
        input_columns = df.columns.astype(str).tolist()
        for column in input_columns:
            if not VARIABLE_PATTERN.match(column):
                while (f"x{ix:03d}" in df.columns) or (f"x{ix:03d}" in (new_columns + input_columns)):
                    ix += 1
                origin_columns.append(column)
                new_columns.append(f"x{ix:03d}")
                ix += 1
            else:
                keep_columns.append(column)
        self.column_mapper_ = dict(zip(origin_columns, new_columns))
        column_mapper = copy(self.column_mapper_)
        column_mapper.update(dict(zip(keep_columns, keep_columns)))
        self.column_mapper = column_mapper
        df.columns = df.columns.map(column_mapper)

    def fit(self, X, y, X_pool: Optional[List[pd.DataFrame]] = None):
        """
        Fits the regression model and returns a new dataframe with the additional features.

        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
            - y: pandas dataframe or numpy array with targets for all n_datapoints
        Returns:
            - new_df: new pandas dataframe with all the original features (except categorical features transformed
                      into multiple 0/1 columns) and the most promising engineered features. This df can then be
                      used to train your final model.

        Please ensure that X only contains valid feature columns (including possible categorical variables).

        Note: we strongly encourage you to name your features X1 ...  Xn or something simple like this before passing
              a DataFrame to this model. This can help avoid potential problems with sympy later on.
              The data should only contain finite values (no NaNs etc.)
        """
        # store column names as they'll be lost in the other check
        cols = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else []
        if self.problem_type is None:
            if type_of_target(y) == "continuous":
                self.problem_type = "regression"
            else:
                self.problem_type = "classification"
        # check input variables
        X, target = check_X_y(X, y, y_numeric=self.problem_type == "regression", dtype=None)
        if self.regularization is None:
            if X.shape[0] > 2000:
                self.regularization = "l2"
            else:
                self.regularization = "l1"
        if not cols:
            # the additional zeros in the name are because of the variable check in _generate_features,
            # where we check if the column name occurs in the the expression. this would lead to many
            # false positives if we have features x1 and x10...x19 instead of x001...x019.
            cols = ["x%03i" % i for i in range(X.shape[1])]
        self.original_columns_ = cols
        # transform X into a dataframe (again)
        pre_df = pd.DataFrame(X, columns=cols)
        # if column_name don't match variable regular-expression-pattern, convert it(keep in mind do same conversion in transform process)
        self.convert_colname_to_variables(pre_df)

        if pre_df.shape[1] > self.max_used_feats:
            # In order to limit the scale of the problem, the number of features is limited to K
            base_model_cls = ExtraTreesClassifier if self.problem_type == "classification" else ExtraTreesRegressor
            base_model_params = dict(
                n_estimators=50,
                min_samples_leaf=10,
                min_samples_split=10,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            feature_importances = base_model_cls(**base_model_params).fit(X, y).feature_importances_
            pre_activated_indexes = np.argsort(-feature_importances)[:self.max_used_feats]
        else:
            pre_activated_indexes = np.arange(pre_df.shape[1])
        boruta = BorutaFeatureSelector(max_depth=7, n_estimators="auto", max_iter=10, weak=False,
                                       random_state=self.random_state, verbose=self.verbose).fit(
            pre_df.values[:, pre_activated_indexes], y)
        if boruta.weak:
            boruta_mask = boruta.support_ + boruta.support_weak_
        else:
            boruta_mask = boruta.support_
        activated_indexes = pre_activated_indexes[boruta_mask]
        df = pre_df.iloc[:, activated_indexes]
        if X_pool:
            X_pool_new = []
            for X_ in X_pool:
                if X_ is None:
                    continue
                if not isinstance(X_, pd.DataFrame):
                    X_ = pd.DataFrame(X_)
                X_ = X_.iloc[:, activated_indexes].copy()
                X_.columns = df.columns
                X_pool_new.append(X_)
            if len(X_pool_new) > 0:
                X_pool = pd.concat(X_pool_new)
                X_pool.index = range(X_pool.shape[0])
            else:
                X_pool = None

        self.boruta_1 = boruta
        self.pre_activated_indexes = pre_activated_indexes
        self.activated_indexes = activated_indexes
        # possibly convert categorical columns
        df = self._transform_categorical_cols(df)
        # if we're not given specific feateng_cols, then just take all columns except categorical
        if self.feateng_cols:
            fcols = []
            for c in self.feateng_cols:
                if c not in self.original_columns_:
                    raise ValueError("[AutoFeat] feateng_col %r not in df.columns" % c)
                if c in self.categorical_cols_map_:
                    fcols.extend(self.categorical_cols_map_[c])
                else:
                    fcols.append(c)
            self.feateng_cols_ = fcols
        else:
            self.feateng_cols_ = list(df.columns)
        # subsample data points and targets in case we'll generate too many features
        # (n_rows * n_cols * 32/8)/1000000000 <= max_gb
        n_cols = n_cols_generated(len(self.feateng_cols_), self.feateng_steps, len(self.transformations))
        n_gb = (len(df) * n_cols) / 250000000
        if self.verbose:
            print("[AutoFeat] The %i step feature engineering process could generate up to %i features." % (
                self.feateng_steps, n_cols))
            print("[AutoFeat] With %i data points this new feature matrix would use about %.2f gb of space." % (
                len(df), n_gb))
        # if self.max_gb and n_gb > self.max_gb:
        #     n_rows = int(self.max_gb * 250000000 / n_cols)
        #     if self.verbose:
        #         print(
        #             "[AutoFeat] As you specified a limit of %.1d gb, the number of data points is subsampled to %i" % (
        #                 self.max_gb, n_rows))
        #     subsample_idx = np.random.permutation(list(df.index))[:n_rows]
        #     df_subs = df.iloc[subsample_idx]
        #     df_subs.reset_index(drop=True, inplace=True)
        #     target_sub = target[subsample_idx]
        # else:
        df_subs = df.copy()
        target_sub = target.copy()
        # generate features
        df_subs, self.feature_formulas_ = engineer_features(
            df_subs,
            self.feateng_cols_,
            None,
            self.feateng_steps,
            self.transformations,
            self.verbose,
            X_pool
        )
        # select predictive features
        self.core_selector = FeatureSelector(self.problem_type, self.featsel_runs, None, self.n_jobs, self.verbose,
                                             self.random_state, self.consider_other, self.regularization)

        if self.featsel_runs <= 0:
            if self.verbose:
                print("[AutoFeat] WARNING: Not performing feature selection.")
            good_cols = df_subs.columns
        else:
            good_cols = self.core_selector.fit(df_subs, target_sub).good_cols_
            # if no features were selected, take the original features
            if not good_cols:
                good_cols = list(df.columns)
        # filter out those columns that were original features or generated otherwise
        self.new_feat_cols_ = [c for c in good_cols if c not in list(df.columns)]
        self.feature_functions_ = {}
        self.good_cols_ = good_cols
        if self.standardize or self.do_final_selection:
            df_final = self._generate_features(pre_df, self.new_feat_cols_)
            if self.do_final_selection:
                boruta = BorutaFeatureSelector(max_depth=7, n_estimators="auto", max_iter=10, weak=False,
                                               random_state=self.random_state, verbose=self.verbose).fit(df_final, y)
                support_mask = boruta.support_
                self.boruta_2 = boruta
                if boruta.weak:
                    support_mask += boruta.support_weak_
                origin_columns = pre_df.columns
                gen_columns = df_final.columns[pre_df.shape[1]:]
                origin_mask = support_mask[:pre_df.shape[1]]
                gen_mask = support_mask[pre_df.shape[1]:]
                gen_valid_cols = gen_columns[gen_mask].tolist()
                self.new_feat_cols_ = [c for c in self.new_feat_cols_ if c in gen_valid_cols]
                origin_valid_cols = origin_columns[origin_mask].tolist()
                self.valid_cols_ = origin_valid_cols + gen_valid_cols
                df_final = df_final[self.valid_cols_]
            else:
                self.valid_cols_ = None
            if self.standardize:
                self.standardizer_ = StandardScaler().fit(df_final)
            else:
                self.standardizer_ = None
        else:
            self.standardizer_ = None
            self.valid_cols_ = None
        return self

    def transform(self, X):
        """
        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
        Returns:
            - new_df: new pandas dataframe with all the original features (except categorical features transformed
                      into multiple 0/1 columns) and the most promising engineered features. This df can then be
                      used to train your final model.
        """
        check_is_fitted(self, ["feature_formulas_"])
        # store column names as they'll be lost in the other check
        cols = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else []
        # check input variables
        X = check_array(X, force_all_finite="allow-nan", dtype=None)
        if not cols:
            cols = ["x%03i" % i for i in range(X.shape[1])]
        if not cols == self.original_columns_:
            raise ValueError("[AutoFeat] Not the same features as when calling fit.")
        # transform X into a dataframe (again)
        df = pd.DataFrame(X, columns=cols)
        # convert_colname_to_variables
        df.columns = df.columns.map(self.column_mapper)
        # possibly convert categorical columns
        df = self._transform_categorical_cols(df)
        # possibly apply pi-theorem
        # generate engineered features
        df = self._generate_features(df, self.new_feat_cols_)
        if self.always_return_numpy:
            return df.to_numpy()
        if self.valid_cols_ is not None:
            df = df[self.valid_cols_]
        if self.standardizer_ is not None:
            df = pd.DataFrame(self.standardizer_.transform(df.values), columns=df.columns, index=df.index)
        # parse inf, nan to median
        inf_cnt = np.count_nonzero(~np.isfinite(df), axis=0)
        if inf_cnt.sum() > 0:
            self.logger.warning(f"inf_cnt.sum() = {inf_cnt.sum()}, "
                                f"error-columns are: {df.columns[inf_cnt > 0].tolist()} , "
                                f"using median-fill handle this")
            data = df.values
            data[~np.isfinite(df)] = np.nan
            data = SimpleImputer(strategy="median").fit_transform(data)  # fixme: 全为0的列
            df = pd.DataFrame(data, columns=df.columns, index=df.index)
        return df

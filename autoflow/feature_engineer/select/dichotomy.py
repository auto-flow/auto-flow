#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from time import time

import category_encoders.utils as util
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target

# from lightgbm import LGBMClassifier, LGBMRegressor
from autoflow.estimator.wrap_lightgbm import LGBMClassifier, LGBMRegressor


class DichotomyFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            base_model="lgbm",
            n_jobs=-1,
            random_state=42,
            max_dichotomy=10,
            cv=3,
            cv_budget=2,
            test_size=0.33
    ):
        self.cv_budget = cv_budget
        self.test_size = test_size
        self.cv = cv
        self.max_dichotomy = max_dichotomy
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.base_model = base_model

    def get_model(self, base_model: str, is_classification):
        lgbm_params = dict(
            boosting_type="gbdt",
            learning_rate=0.01,
            max_depth=31,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            random_state=self.random_state,
            lambda_l1=0.1,
            lambda_l2=0.2,
            subsample_for_bin=40000,
            min_child_weight=0.01,
            verbose=-1,
            n_jobs=self.n_jobs,
            n_estimators=10
        )
        rf_params = dict(
            n_estimators=10
        )
        if base_model == "lgbm":
            if is_classification:
                return LGBMClassifier(**lgbm_params)
            else:
                return LGBMRegressor(**lgbm_params)
        elif base_model == "rf":
            if is_classification:
                return RandomForestClassifier(**rf_params)
            else:
                return RandomForestRegressor(**rf_params)
        elif base_model == "et":
            if is_classification:
                return ExtraTreesClassifier(**rf_params)
            else:
                return ExtraTreesRegressor(**rf_params)
        else:
            raise ValueError(f"Unknown base_model {base_model}")

    def get_support_mask(self, support_list, M):
        if len(support_list) == 0:
            return np.ones([M], dtype="bool")
        elif len(support_list) == 1:
            return support_list[0]
        else:
            L = len(support_list)
            cur_support = None
            for i in range(L - 1, 1, -1):
                if cur_support is None:
                    cur_support = support_list[i]
                prev_support = support_list[i - 1]
                logical_and = np.logical_and(cur_support, prev_support)
                if np.count_nonzero(logical_and):
                    cur_support = logical_and
                else:
                    cur_support = np.logical_or(cur_support, prev_support)
            return cur_support

    def fit(self, X, y):
        X = util.convert_input(X)
        X_ = X.values.copy()
        y = check_array(y, ensure_2d=False, dtype="float")
        y = np.array(y)
        N, M = X.shape
        target_type = type_of_target(y)
        self.is_classification = (target_type != "continuous")
        if self.is_classification:
            self.kfold = StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        else:
            self.kfold = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        score_list = []
        feature_importance_list = []
        support_list = []
        for dichotomy_cnt in range(self.max_dichotomy):
            feature_importances = []
            scores = []
            cv_start_time = time()
            if dichotomy_cnt > 0:
                support_mask = support_list[-1]
            else:
                support_mask = np.ones([M], dtype="bool")
            support_cols = np.arange(M)[support_mask]
            for train_ix, valid_ix in self.kfold.split(X_, y):
                X_train = X_[train_ix, :][:, support_mask]
                y_train = y[train_ix]
                X_valid = X_[valid_ix, :][:, support_mask]
                y_valid = y[valid_ix]
                model = self.get_model(self.base_model, self.is_classification)
                model.fit(X_train, y_train)
                feature_importance = model.model.feature_importance("gain")  # todo: 适配
                feature_importances.append(feature_importance)
                score = model.score(X_valid, y_valid)
                scores.append(score)
                if time() - cv_start_time > self.cv_budget:
                    break
            cv_cost_time = time() - cv_start_time
            feature_importance = np.vstack(feature_importances).mean(axis=0)
            score = np.mean(scores)
            if dichotomy_cnt > 0:
                L = len(support_list)
                support_mask_cnt = np.count_nonzero(support_mask)
                if score >= max(score_list):
                    # 表现好， 继续减半
                    # 向上找到第一个小于support_mask_cnt
                    lower_mask_cnt = 0
                    for i in range(L - 2, -1, -1):
                        cnt = np.count_nonzero(support_list[i])
                        if cnt < support_mask_cnt:
                            lower_mask_cnt = cnt
                    kept_features = lower_mask_cnt + (support_mask_cnt - lower_mask_cnt) // 2
                    upper_feature_importance = feature_importance
                    upper_support_mask = support_mask
                else:
                    # 表现不好，扩增特征，向上找到第一个与support_list[-1]有交集的support

                    upper_i = None
                    for i in range(L - 2, -1, -1):
                        if np.count_nonzero(support_list[i]) > support_mask_cnt:
                            upper_i = i
                            break
                    if upper_i is not None:
                        upper_support_mask = support_list[upper_i]
                        upper_feature_importance = feature_importance_list[upper_i + 1]
                    else:
                        upper_support_mask = np.ones([M], dtype="bool")
                        upper_feature_importance = feature_importance_list[0] # todo assert true
                    cur_features = np.count_nonzero(support_mask)
                    kept_features = (np.count_nonzero(upper_support_mask) - cur_features) // 2 + cur_features
                upper_support_index = np.arange(M)[upper_support_mask]
                index = np.argsort(-upper_feature_importance)[:kept_features]
                new_support_index = upper_support_index[index]
                new_support_mask = np.zeros([M], dtype="bool")
                new_support_mask[new_support_index] = True
                support_list.append(new_support_mask)
            else:
                # 首次 直接减半
                index = np.argsort(-feature_importance)[:M // 2]
                support = np.zeros([M], dtype="bool")
                support[index] = True
                support_list.append(support)
            feature_importance_list.append(feature_importance)
            score_list.append(score)

        return self


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from autoflow.utils.logging_ import setup_logger
    from sklearn.model_selection import train_test_split

    setup_logger()
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    selector = DichotomyFeatureSelector()
    selector.fit(X, y)

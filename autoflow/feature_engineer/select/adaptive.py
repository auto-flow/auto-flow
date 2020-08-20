#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from time import time

import category_encoders.utils as util
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target

from autoflow.estimator.wrap_lightgbm import LGBMClassifier, LGBMRegressor
from autoflow.utils.logging_ import get_logger


class AdaptiveFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            percentage=20,
            feats_must_less_than_rows=True,
            lgbm_w=0.5,
            et_iters=100,
            lgbm_iters=100,
            et_budget=1.5,
            lgbm_budget=1.5,
            step=10,
            n_jobs=-1,
            random_state=42

    ):
        self.feats_must_less_than_rows = feats_must_less_than_rows
        self.random_state = random_state
        self.lgbm_w = lgbm_w
        self.lgbm_budget = lgbm_budget
        self.et_budget = et_budget
        self.n_jobs = n_jobs
        self.step = step
        self.lgbm_iters = lgbm_iters
        self.et_iters = et_iters
        self.percentage = float(np.clip(percentage, 0, 100))
        self.logger = get_logger(self)

    ok_msg = "'s all iterations are completed."
    es_msg = " early stopped."

    def fit(self, X, y):
        X = util.convert_input(X)
        y = check_array(y, ensure_2d=False, dtype="float")
        N, M = X.shape
        target_type = type_of_target(y)
        self.is_classification = (target_type != "continuous")
        self.lgbm, self.lgbm_cost_time, self.lgbm_imp = self.fit_model(
            LGBMClassifier,
            LGBMRegressor,
            dict(),
            self.lgbm_budget,
            self.lgbm_iters,
            X.values,
            y,
            lambda lgbm: lgbm.model.feature_importance("gain"),
            self.lgbm_w
        )
        self.et, self.lgbm_cost_time, self.et_imp = self.fit_model(
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            dict(max_depth=10),
            self.et_budget,
            self.et_iters,
            X.values,
            y,
            lambda et: et.feature_importances_,
            1 - self.lgbm_w
        )
        imp_vecs = [self.lgbm_imp, self.et_imp]
        ws = [self.lgbm_w, 1 - self.lgbm_w]
        feature_importance = np.zeros([M])
        for imp_vec, w in zip(imp_vecs, ws):
            feature_importance += w * imp_vec
        self.feature_importance = feature_importance
        if self.feats_must_less_than_rows:
            max_feats = min(N, M)
        else:
            max_feats = M
        min_feats = 1
        n_kept_feats = min_feats + round((max_feats - min_feats) * (self.percentage) / 100)
        self.logger.info(f"min_feats = {min_feats}, max_feats = {max_feats}, "
                         f"percentage = {self.percentage}, n_kept_feats = {n_kept_feats}")
        index = np.argsort(-self.feature_importance)[:n_kept_feats]
        self.index = np.sort(index)
        columns = np.array(X.columns)
        self.columns = columns[self.index]
        return self

    def transform(self, X):
        X = util.convert_input(X)
        return X[self.columns]

    def fit_model(self, clf_cls, reg_cls, params, budget, max_iters, X, y, feat_imp_callback, weight):
        N, M = X.shape
        if budget == 0 or weight == 0:
            return None, 0, np.zeros([M])
        default_params = {
            "n_estimators": self.step,
            "n_jobs": self.n_jobs,
            "warm_start": True,
            "random_state": self.random_state
        }
        default_params.update(params)
        if self.is_classification:
            model = clf_cls(**default_params)
        else:
            model = reg_cls(**default_params)

        start_time = time()
        msg = "'s all iterations are completed."
        for iter in range(self.step, max_iters + self.step, self.step):
            model.n_estimators = iter
            model.fit(X, y)
            if time() - start_time > budget:
                msg = " early stopped."
                break
        cost_time = time() - start_time

        self.logger.info(f"{model.__class__.__name__}{msg} max_iters = {model.n_estimators}, "
                         f"budget = {budget:.2f}, cost_time = {cost_time:.2f} .")

        feature_importance = feat_imp_callback(model)
        scaler = MinMaxScaler()
        feature_importance_ = scaler.fit_transform(feature_importance[:, None]).flatten()

        return model, cost_time, feature_importance_


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from autoflow.utils.logging_ import setup_logger
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    setup_logger()
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    pipe = Pipeline([
        ("selector", AdaptiveFeatureSelector(percentage=50)),
        ("lgbm", LGBMClassifier(n_estimators=500))
    ])
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    print(score)

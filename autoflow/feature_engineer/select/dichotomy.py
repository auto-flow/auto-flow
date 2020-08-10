#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from collections import Counter
from time import time

import category_encoders.utils as util
import numpy as np
from frozendict import frozendict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier, Lasso
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import check_array
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.multiclass import type_of_target

# from lightgbm import LGBMClassifier, LGBMRegressor
from autoflow.estimator.wrap_lightgbm import LGBMClassifier, LGBMRegressor
from autoflow.utils.logging_ import get_logger


class DichotomyFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            base_model="lgbm",
            n_jobs=-1,
            random_state=42,
            max_dichotomy=10,
            cv=3,
            cv_budget=2,
            test_size=0.33,
            model_params=frozendict()
    ):
        self.model_params = model_params
        self.cv_budget = cv_budget
        self.test_size = test_size
        self.cv = cv
        self.max_dichotomy = max_dichotomy
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.base_model = base_model
        self.logger = get_logger(self)

    def get_model(self):
        lgbm_params = dict(
            boosting_type="gbdt",
            learning_rate=0.1,
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
            n_estimators=10,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            min_samples_leaf=10,
            min_samples_split=10
        )
        ridge_params = dict(
            normalize=True, random_state=self.random_state
        )
        lasso_params = dict(
            normalize=True, random_state=self.random_state
        )
        lr_params = dict(penalty="l1", solver="saga", C=0.1)
        if self.model_params:
            for params in (lgbm_params, rf_params, ridge_params, lasso_params, lr_params):
                params.update(self.model_params)
        base_model = self.base_model
        is_classification = self.is_classification
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
        elif base_model == "ridge":
            if is_classification:
                return RidgeClassifier(**ridge_params)
            else:
                return Ridge(**ridge_params)
        elif base_model == "lasso":
            if is_classification:
                return LogisticRegression(**lr_params)
            else:
                return Lasso(**lasso_params)
        else:
            raise ValueError(f"Unknown base_model {base_model}")

    def get_feature_importance(self, model):
        if self.base_model == "lgbm":
            return model.model.feature_importance("gain")
        elif self.base_model in ("rf", "et"):
            return model.feature_importances_
        elif self.base_model in ("ridge", "lasso"):
            coef = model.coef_
            if np.ndim(coef) == 1:
                return np.abs(model.coef_)
            else:
                return np.abs(model.coef_).sum(axis=0)
        else:
            raise ValueError(f"Unknown base_model {self.base_model}")

    @ignore_warnings(category=ConvergenceWarning)
    def evaluate(self, prev_support_mask, prev_feature_importance, n_feats, X, y):
        prev_support_index = np.arange(X.shape[1])[prev_support_mask]
        support_index = prev_support_index[np.argsort(-prev_feature_importance)[:n_feats]]
        support_mask = np.zeros([X.shape[1]], dtype="bool")
        support_mask[support_index] = True

        cv_start_time = time()
        feature_importances = []
        scores = []
        cv_cnt = 0
        for train_ix, valid_ix in self.kfold.split(X, y):
            X_train = X[train_ix, :][:, support_mask]
            y_train = y[train_ix]
            X_valid = X[valid_ix, :][:, support_mask]
            y_valid = y[valid_ix]
            model = self.get_model()
            model.fit(X_train, y_train)
            feature_importance = self.get_feature_importance(model)
            feature_importances.append(feature_importance)
            score = model.score(X_valid, y_valid)
            scores.append(score)
            cv_cnt += 1
            if time() - cv_start_time > self.cv_budget:
                break
        cv_cost_time = time() - cv_start_time
        feature_importance = np.vstack(feature_importances).mean(axis=0)
        score = np.mean(scores)
        return {
            "score": score,
            "feature_importance": feature_importance,
            "support_mask": support_mask,
            "cv_cnt": cv_cnt,
            "cv_cost_time": cv_cost_time
        }

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
        feat_mapper = {}
        feat_mapper[M] = self.evaluate(np.ones([M], dtype="bool"), np.zeros([M]), M, X_, y)
        for dichotomy_cnt in range(self.max_dichotomy):
            # 1. 在最好的两边选值
            if len(feat_mapper) == 1:
                feat1 = round(M / 3)
                feat2 = feat1 * 2
                cand_feats = [feat1, feat2]
                prev_feats = [M, M]
            else:
                feats = sorted(list(feat_mapper.keys()))
                scores = [feat_mapper[feat]["score"] for feat in feats]
                best_feat_ix = int(np.argmax(scores))  # todo: 倾向于用少特征
                cand_feats = []
                prev_feats = []
                # left
                if best_feat_ix == 0:
                    cand_feats.append(feats[best_feat_ix] // 2)
                else:
                    cand_feats.append(
                        round(feats[best_feat_ix - 1] + (feats[best_feat_ix] - feats[best_feat_ix - 1]) / 2))
                prev_feats.append(feats[best_feat_ix])
                # right
                if best_feat_ix != len(feats) - 1:
                    cand_feats.append(
                        round(feats[best_feat_ix] + (feats[best_feat_ix + 1] - feats[best_feat_ix]) / 2))
                    prev_feats.append(feats[best_feat_ix + 1])
                else:
                    self.logger.debug(f"best_feat_ix = {best_feat_ix}, equals `len(feats) - 1)`")
            should_break = False
            for feat in cand_feats:
                if feat in feat_mapper:
                    should_break = True
                    break
                if feat <= 0:
                    should_break = True
                    break
            if should_break:
                break
            pairs = list(zip(cand_feats, prev_feats))
            for cand_feat, prev_feat in set(pairs):
                feat_mapper[cand_feat] = self.evaluate(
                    feat_mapper[prev_feat]["support_mask"],
                    feat_mapper[prev_feat]["feature_importance"],
                    cand_feat,
                    X_, y
                )
        self.feat_mapper = feat_mapper
        feats = sorted(list(feat_mapper.keys()))
        scores = [feat_mapper[feat]["score"] for feat in feats]
        best_feat_ix = int(np.argmax(scores))  # todo: 倾向于用少特征
        best_feat = feats[best_feat_ix]
        self.best_score = scores[best_feat_ix]
        self.best_feat = best_feat
        self.support_mask = self.feat_mapper[best_feat]["support_mask"]
        self.columns = X.columns[self.support_mask]
        msg0 = f"After {len(feat_mapper)} times evaluations, " \
            f"{M - best_feat} features are filtered, " \
            f"{best_feat} features are preserved({(best_feat / M) * 100:.2f}%), " \
            f"best evaluation score = {self.best_score:.3f}"
        self.logger.info(msg0)
        cv_cnts = [feat_mapper[key]["cv_cnt"] for key in feat_mapper]
        cv_cost_times = [feat_mapper[key]["cv_cost_time"] for key in feat_mapper]
        cv_cnt_counter = Counter(cv_cnts)
        msg1 = "'cv_cnt' statistics: "
        for cv_cnt, times in cv_cnt_counter.items():
            msg1 += f"{cv_cnt}-cv [{times} times]; "
        self.logger.info(msg1)
        msg2 = f"'cv_cost_times' statistics: mean = {np.mean(cv_cost_times):.2f}; " \
            f"max = {np.max(cv_cost_times):.2f}; " \
            f"min = {np.min(cv_cost_times):.2f}; " \
            f"sum = {np.sum(cv_cost_times):.2f}; " \
            f"cv_budget = {self.cv_budget}; "
        self.logger.info(msg2)
        self.msg = "\n".join([msg0, msg1, msg2])
        return self

    def transform(self, X):
        X = util.convert_input(X)
        return X[self.columns]


if __name__ == '__main__':
    from autoflow.utils.logging_ import setup_logger
    from sklearn.pipeline import Pipeline
    from copy import deepcopy

    setup_logger()
    # X, y = load_digits(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # selector = DichotomyFeatureSelector()
    # selector.fit(X, y)
    from joblib import load

    df = load("/home/tqc/PycharmProjects/autoflow/data/2198.bz2")
    df.pop("Name")
    df.pop("Smiles")
    df.pop("labels")
    y = np.array(df.pop("pIC50"))
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    pipeline = Pipeline([
        ("selector", DichotomyFeatureSelector(base_model="lgbm")),
        # ("ridge", Ridge(random_state=0, normalize=True)),
        ("lgbm", LGBMRegressor()),
    ])
    scores = []
    models = []
    for train_ix, valid_ix in cv.split(df, y):
        pipeline.fit(df.iloc[train_ix, :], y[train_ix])
        score = pipeline.score(df.iloc[valid_ix, :], y[valid_ix])
        scores.append(score)
        models.append(deepcopy(pipeline))
        print(score)
    # scores = cross_val_score(pipeline, df, y, cv=cv)
    print(scores)

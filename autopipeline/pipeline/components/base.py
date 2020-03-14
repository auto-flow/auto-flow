import inspect
from copy import deepcopy
from importlib import import_module
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from autopipeline.pipeline.dataframe import GeneralDataFrame
from autopipeline.utils.data import densify


class AutoPLComponent(BaseEstimator):
    cls_hyperparams: dict = {}
    module__ = None
    class__ = None
    classification_only = False
    regression_only = False

    def __init__(self):
        self.estimator = None
        self.hyperparams = deepcopy(self.cls_hyperparams)
        self.set_params(**self.hyperparams)
        self.in_feat_grp = None
        self.out_feat_grp = None

    # @classmethod
    @property
    def class_(cls):
        if not cls.class__:
            raise NotImplementedError()
        return cls.class__

    # @classmethod
    @property
    def module_(self):
        if not self.module__:
            raise NotImplementedError()
        return self.module__

    def get_estimator_class(self):
        M = import_module(self.module_)
        return getattr(M, self.class_)

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = deepcopy(hyperparams)
        should_pop = []
        updated = {}
        for key, value in hyperparams.items():
            key: str
            if key.startswith("_") and (not key.startswith("__")):
                should_pop.append(key)
                key = key[1:]
                new_key, indicator = key.split("-")
                updated[new_key] = self.do_process(indicator, hyperparams, value)
        for key in should_pop:
            hyperparams.pop(key)
        hyperparams.update(updated)
        return hyperparams

    def after_process_estimator(self, estimator, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None,
                                y_test=None):
        return estimator

    def before_fit_X(self, X):
        return X

    def before_fit_y(self, y):
        return y

    def before_pred_X(self, X):
        return X

    def after_pred_X(self, X):
        return X

    def after_pred(self, y):
        return y

    def _pred_or_trans(self, X_train_, X_valid_=None, X_test_=None, X_train=None, X_valid=None, X_test=None,
                       is_train=False):
        raise NotImplementedError

    def pred_or_trans(self, X_train, X_valid=None, X_test=None, is_train=False):
        X_train_ = self.preprocess_data(X_train)
        X_valid_ = self.preprocess_data(X_valid)
        X_test_ = self.preprocess_data(X_test)
        if not self.estimator:
            raise NotImplementedError()
        return self._pred_or_trans(X_train_, X_valid_, X_test_, X_train, X_valid, X_test, is_train)

    def filter_invalid(self, cls, hyperparams: Dict) -> Dict:
        validated = {}
        for key, value in hyperparams.items():
            if key in inspect.signature(cls.__init__).parameters.keys():
                validated[key] = value
            else:
                pass
        return validated

    @staticmethod
    def get_properties():
        raise NotImplementedError()

    def preprocess_data(self, X: Optional[GeneralDataFrame]):
        # todo 考虑在这里多densify
        if X is None:
            return None
        elif isinstance(X, GeneralDataFrame):
            return X.filter_feat_grp(self.in_feat_grp).values
        elif isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise NotImplementedError

    def fit(self, X_train, y_train=None,
            X_valid=None, y_valid=None,
            X_test=None, y_test=None):
        # 只选择当前需要的feat_grp
        X_train_ = self.preprocess_data(X_train)
        X_valid_ = self.preprocess_data(X_valid)
        X_test_ = self.preprocess_data(X_test)
        # 通过以上步骤，保证所有的X都是np.ndarray 形式
        self.shape = X_train_.shape
        # 默认采用代理模式（但可以颠覆这种模式，完全重写这个类）
        cls = self.get_estimator_class()
        # 根据构造函数构造代理估计器
        self.estimator = cls(
            **self.filter_invalid(
                cls, self.after_process_hyperparams(self.hyperparams)
            )
        )
        # 对数据进行预处理（比如有的preprocessor只能处理X>0的数据）
        X_train_ = self.before_fit_X(X_train_)
        y_train = self.before_fit_y(y_train)
        X_test_ = self.before_fit_X(X_test_)
        y_test = self.before_fit_y(y_test)
        X_valid_ = self.before_fit_X(X_valid_)
        y_valid = self.before_fit_y(y_valid)
        # 对代理的estimator进行预处理
        self.estimator = self.after_process_estimator(self.estimator, X_train_, y_train, X_valid_, y_valid, X_test_,
                                                      y_test)
        # todo:  根据原信息判断是否要densify
        X_train_ = densify(X_train_)
        X_valid_ = densify(X_valid_)
        X_test_ = densify(X_test_)
        self._fit(self.estimator, X_train_, y_train, X_valid_, y_valid, X_test_,
                  y_test)

        return self

    def prepare_X_to_fit(self, X_train, X_valid=None, X_test=None):
        return X_train

    def _fit(self, estimator, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None,
             y_test=None):
        # 保留其他数据集的参数，方便模型拓展
        estimator.fit(self.prepare_X_to_fit(X_train, X_valid, X_test), y_train)

    def set_addition_info(self, dict_: dict):
        for key, value in dict_.items():
            setattr(self, key, value)

    def update_hyperparams(self, hp: dict):
        '''set default hyperparameters in init'''
        self.hyperparams.update(hp)
        # self.set_params(**self.hyperparams)

    def get_estimator(self):
        return self.estimator

    def do_process(self, indicator, hyperparams, value):
        if indicator == "lr_ratio":
            lr = hyperparams["learning_rate"]
            return max(int(value * (1 / lr)), 10)
        elif indicator == "sp1_ratio":
            if hasattr(self, "shape"):
                n_components = max(
                    int(self.shape[1] * value),
                    1
                )
            else:
                print("warn")
                n_components = 100
            return n_components
        else:
            raise NotImplementedError()

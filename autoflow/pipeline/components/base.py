import inspect
import math
import pickle
from copy import deepcopy
from importlib import import_module
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from autoflow.pipeline.dataframe import GenericDataFrame
from autoflow.utils.data import densify
from autoflow.utils.dataframe import rectify_dtypes
from autoflow.utils.hash import get_hash_of_Xy, get_hash_of_dict
from autoflow.utils.logging import get_logger


class AutoFlowComponent(BaseEstimator):
    module__ = None
    class__ = None
    classification_only = False
    regression_only = False
    boost_model = False
    tree_model = False
    store_intermediate = False
    suspend_other_processes = False
    is_fit = False

    def __init__(self):
        self.resource_manager = None
        self.estimator = None
        self.in_feature_groups = None
        self.out_feature_groups = None
        self.hyperparams = {}
        self.logger = get_logger(self)

    def _get_param_names(cls):
        return sorted(cls.hyperparams.keys())

    @property
    def class_(self):
        if not self.class__:
            raise NotImplementedError()
        return self.class__

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
        hyperparams = self.before_parse_escape_hyperparameters(hyperparams)
        should_pop = []
        updated = {}
        for key, value in hyperparams.items():
            key: str
            if key.startswith("_") and (not key.startswith("__")):
                should_pop.append(key)
                key = key[1:]
                new_key, indicator = key.split("-")
                updated[new_key] = self.parse_escape_hyperparameters(indicator, hyperparams, value)
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

    def _pred_or_trans(self, X_train_, X_valid_=None, X_test_=None, X_train=None, X_valid=None, X_test=None,
                       y_train=None):
        raise NotImplementedError

    def pred_or_trans(self, X_train, X_valid=None, X_test=None, y_train=None):
        X_train_ = self.preprocess_data(X_train)
        X_valid_ = self.preprocess_data(X_valid)
        X_test_ = self.preprocess_data(X_test)
        if not self.estimator:
            raise NotImplementedError()
        return self._pred_or_trans(X_train_, X_valid_, X_test_, X_train, X_valid, X_test, y_train)

    def filter_invalid(self, cls, hyperparams: Dict) -> Dict:
        hyperparams = deepcopy(hyperparams)
        validated = {}
        for key, value in hyperparams.items():
            if key in inspect.signature(cls.__init__).parameters.keys():
                validated[key] = value
            else:
                pass
        return validated

    def preprocess_data(self, X: Optional[GenericDataFrame], extract_info=False):
        # todo 考虑在这                                                                                                                                                                                    里多densify
        if X is None:
            return None
        elif isinstance(X, GenericDataFrame):
            from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm
            if issubclass(self.__class__, AutoFlowFeatureEngineerAlgorithm):
                df = X.filter_feature_groups(self.in_feature_groups)
            else:
                df = X
            rectify_dtypes(df)
            if extract_info:
                return df, df.feature_groups, df.columns_metadata
            else:
                return df
        elif isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise NotImplementedError

    def build_proxy_estimator(self):
        # 默认采用代理模式（但可以颠覆这种模式，完全重写这个类）
        cls = self.get_estimator_class()
        # 根据构造函数构造代理估计器
        self.processed_params = self.filter_invalid(
            cls, self.after_process_hyperparams(self.hyperparams)
        )
        self.estimator = cls(
            **self.processed_params
        )

    def fit(self, X_train, y_train=None,
            X_valid=None, y_valid=None,
            X_test=None, y_test=None):
        # todo: sklearn 对于 DataFrame 是支持的， 是否需要修改？
        # 只选择当前需要的feature_groups
        assert isinstance(X_train, GenericDataFrame)
        X_train_, feature_groups, columns_metadata = self.preprocess_data(X_train, True)
        X_valid_ = self.preprocess_data(X_valid)
        X_test_ = self.preprocess_data(X_test)
        # 通过以上步骤，保证所有的X都是np.ndarray 形式
        self.shape = X_train_.shape
        self.build_proxy_estimator()
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
        # todo: 测试特征全部删除的情况
        if len(X_train_.shape) > 1 and X_train_.shape[1] > 0:
            self.estimator = self._fit(self.estimator, X_train_, y_train, X_valid_, y_valid, X_test_,
                                       y_test, feature_groups, columns_metadata)
            self.is_fit = True
        else:
            self.logger.warning(
                f"Component: {self.__class__.__name__} is fitting a empty data.\nShape of X_train_ = {X_train_.shape}.")
        return self

    def prepare_X_to_fit(self, X_train, X_valid=None, X_test=None):
        return X_train

    def _fit(self, estimator, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None,
             y_test=None, feature_groups=None, columns_metadata=None):
        # 保留其他数据集的参数，方便模型拓展
        X = self.prepare_X_to_fit(X_train, X_valid, X_test)
        if self.store_intermediate:
            if self.resource_manager is None:
                print("warn: no resource_manager when store_intermediate is True")
                fitted_estimator = self.core_fit(estimator, X, y_train, X_valid, y_valid, X_test, y_test,
                                                 feature_groups, columns_metadata)
            else:
                # get hash value from X, y, hyperparameters
                Xy_hash = get_hash_of_Xy(X, y_train)
                hp_hash = get_hash_of_dict(self.processed_params)
                hash_value = Xy_hash + "-" + hp_hash
                result = self.resource_manager.redis_get(hash_value)
                if result is None:
                    fitted_estimator = estimator.fit(X, y_train)
                    self.resource_manager.redis_set(hash_value, pickle.dumps(fitted_estimator))
                else:
                    fitted_estimator = pickle.loads(result)
        else:
            fitted_estimator = self.core_fit(estimator, X, y_train, X_valid, y_valid, X_test, y_test, feature_groups,
                                             columns_metadata)
        self.resource_manager = None  # avoid can not pickle error
        return fitted_estimator

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None, columns_metadata=None):
        return estimator.fit(X, y)

    def set_addition_info(self, dict_: dict):
        for key, value in dict_.items():
            setattr(self, key, value)

    def update_hyperparams(self, hp: dict):
        '''set default hyperparameters in init'''
        self.hyperparams.update(hp)
        self.set_addition_info(hp)

    def get_estimator(self):
        return self.estimator

    def before_parse_escape_hyperparameters(self, hyperparams):
        return hyperparams

    def parse_escape_hyperparameters(self, indicator, hyperparams, value):
        if indicator == "lr_ratio":
            lr = hyperparams["learning_rate"]
            return max(int(value * (1 / lr)), 10)
        elif indicator == "sp1_ratio":
            factor = "shape"
            if hasattr(self, factor):
                n_components = max(
                    int(self.shape[1] * value),
                    1
                )
            else:
                self.logger.warning(f"{str(self)} haven't attribute {factor}")
                n_components = 100
            return n_components
        elif indicator == "sp1_percent":
            factor = "shape"
            if hasattr(self, factor):
                n_components = max(
                    int(self.shape[1] * (value / 100)),
                    1
                )
            else:
                self.logger.warning(f"{str(self)} haven't attribute {factor}")
                n_components = 100
            return n_components
        elif indicator == "sp1_dev":
            factor = "shape"
            if hasattr(self, factor):
                if value == 0:
                    value = 1
                n_components = max(
                    math.ceil(self.shape[1] / value),
                    1
                )
            else:
                self.logger.warning(f"{str(self)} haven't attribute {factor}")
                n_components = 100
            return n_components
        elif indicator == "card_ratio":
            factor = "cardinality"
            if hasattr(self, factor):
                n_components = max(
                    math.ceil(self.cardinality * value),
                    2
                )
            else:
                self.logger.warning(f"{str(self)} haven't attribute {factor}")
                n_components = 6
            return n_components
        else:
            raise NotImplementedError()

    def before_pred_X(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return X

import inspect
import math
from copy import deepcopy
from importlib import import_module
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from autoflow.data_container import DataFrameContainer
from autoflow.data_container import NdArrayContainer
from autoflow.data_container.base import DataContainer
from autoflow.utils.logging_ import get_logger


class AutoFlowComponent(BaseEstimator):
    module__ = None
    class__ = None
    classification_only = False
    regression_only = False
    boost_model = False
    tree_model = False
    cache_intermediate = False
    support_early_stopping = False
    is_fit = False
    additional_info_keys = tuple()

    def __init__(self, **kwargs):
        self.resource_manager = None
        self.component = None
        self.in_feature_groups = None
        self.out_feature_groups = None
        self.hyperparams = kwargs
        self.set_inside_dict(kwargs)
        self.logger = get_logger(self)

    def update_hyperparams(self, kwargs):
        self.hyperparams.update(kwargs)
        self.set_inside_dict(kwargs)

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
                new_key, indicator = key.split("__")
                updated[new_key] = self.parse_escape_hyperparameters(indicator, hyperparams, value)
        for key in should_pop:
            hyperparams.pop(key)
        hyperparams.update(updated)
        return hyperparams

    def after_process_estimator(self, estimator, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None,
                                y_test=None):
        return estimator

    def before_fit_X(self, X: DataFrameContainer):
        if X is None:
            return None
        if isinstance(X, DataContainer):
            return X.data
        return X

    def before_fit_y(self, y: NdArrayContainer):
        if y is None:
            return None
        return y.data

    def filter_invalid(self, cls, hyperparams: Dict) -> Dict:
        hyperparams = deepcopy(hyperparams)
        origin_hp_set = set(hyperparams.keys())
        validated = {}
        for key, value in hyperparams.items():
            if key in inspect.signature(cls.__init__).parameters.keys():
                validated[key] = value
            else:
                pass
        current_hp_set = set(validated.keys())
        diff = origin_hp_set - current_hp_set
        if diff:
            self.logger.debug(f"{list(diff)} are filtered in {self.__class__.__name__}")
        return validated

    def filter_feature_groups(self, X: Optional[DataFrameContainer]):
        if X is None:
            return None
        assert isinstance(X, DataFrameContainer)
        from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm
        if issubclass(self.__class__, AutoFlowFeatureEngineerAlgorithm):
            df = X.filter_feature_groups(self.in_feature_groups)
        else:
            df = X
        # rectify_dtypes(df)
        return df

    def build_proxy_estimator(self):
        if self.component is not None:
            return
        # 默认采用代理模式（但可以颠覆这种模式，完全重写这个类）
        cls = self.get_estimator_class()
        # 根据构造函数构造代理估计器
        self.processed_params = self.filter_invalid(
            cls, self.after_process_hyperparams(self.hyperparams)
        )
        self.component = cls(
            **self.processed_params
        )

    def fit(self, X_train, y_train=None,
            X_valid=None, y_valid=None,
            X_test=None, y_test=None):
        # 只选择当前需要的feature_groups
        assert isinstance(X_train, DataFrameContainer)
        X_train = self.filter_feature_groups(X_train)
        X_valid = self.filter_feature_groups(X_valid)
        X_test = self.filter_feature_groups(X_test)
        self.shape = X_train.shape
        self.build_proxy_estimator()
        feature_groups = X_train.feature_groups
        # 对数据进行预处理（比如有的preprocessor只能处理X>0的数据）
        X_train_ = self.before_fit_X(X_train)
        y_train_ = self.before_fit_y(y_train)
        X_test_ = self.before_fit_X(X_test)
        y_test_ = self.before_fit_y(y_test)
        X_valid_ = self.before_fit_X(X_valid)
        y_valid_ = self.before_fit_y(y_valid)
        # 对代理的estimator进行预处理
        if not self.is_fit:
            self.component = self.after_process_estimator(self.component, X_train_, y_train_, X_valid_,
                                                          y_valid_, X_test_, y_test_)
        # todo: 测试特征全部删除的情况
        if len(X_train.shape) > 1 and X_train.shape[1] > 0:
            self.component = self._fit(self.component, X_train_, y_train_, X_valid_,
                                       y_valid_, X_test_, y_test_, feature_groups)
            self.is_fit = True
        else:
            self.logger.warning(
                f"Component: {self.__class__.__name__} is fitting a empty data.\nShape of X_train_ = {X_train.shape}.")
        return self

    def prepare_X_to_fit(self, X_train, X_valid=None, X_test=None, **kwargs) -> pd.DataFrame:
        return self.before_fit_X(X_train)

    def _fit(self, estimator, X_train, y_train=None, X_valid=None,
             y_valid=None, X_test=None, y_test=None, feature_groups=None):
        # 保留其他数据集的参数，方便模型拓展
        X = self.prepare_X_to_fit(X_train, X_valid, X_test)
        kwargs = {}
        if hasattr(self, "sample_weight"):
            if isinstance(self.sample_weight, np.ndarray) and self.sample_weight.shape[0] == y_train.shape[0]:
                kwargs.update({"sample_weight": self.sample_weight})
            else:
                self.logger.warning(f"Invalid sample_weight. ")
        fitted_estimator = self.core_fit(estimator, X, y_train, X_valid, y_valid, X_test, y_test, feature_groups,
                                         **kwargs)
        return fitted_estimator

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None, **kwargs):
        return estimator.fit(X, y, **kwargs)

    def set_inside_dict(self, dict_: dict):
        for key, value in dict_.items():
            setattr(self, key, value)

    def get_estimator(self):
        return self.component

    def before_parse_escape_hyperparameters(self, hyperparams):
        return hyperparams

    def parse_escape_hyperparameters(self, indicator, hyperparams, value):
        if indicator == "lr_ratio":
            lr = hyperparams["learning_rate"]
            return max(int(value * (1 / lr)), 10)
        elif indicator == "sp1_ratio":
            factor = "shape"
            if hasattr(self, factor):
                n_components = max(1, min(self.shape[0], round(self.shape[1] * value)))
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

    def before_pred_X(self, X: DataFrameContainer):
        return X.data

    @property
    def additional_info(self):
        return dict(self.form_additional_info_pair(key) for key in self.additional_info_keys)

    def form_additional_info_pair(self, key):
        return (
            key,
            getattr(self.component, key, None)
        )

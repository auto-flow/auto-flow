# -*- encoding: utf-8 -*-
from copy import deepcopy
from typing import Union, Any, Dict, Sequence

import numpy as np
import pandas as pd

from hyperflow.pipeline.components.utils import stack_Xs
from hyperflow.pipeline.dataframe import GenericDataFrame
from hyperflow.utils.data import is_nan, is_cat, is_highR_nan
from hyperflow.utils.dataframe import pop_if_exists
from hyperflow.utils.klass import StrSignatureMixin
from hyperflow.utils.logging import get_logger
from hyperflow.utils.ml_task import MLTask, get_ml_task_from_y


class DataManager(StrSignatureMixin):

    def __init__(
            self,
            X_train: Union[pd.DataFrame, GenericDataFrame, np.ndarray, None] = None,
            y_train: Union[pd.Series, np.ndarray, str, None] = None,
            X_test: Union[pd.DataFrame, GenericDataFrame, np.ndarray, None] = None,
            y_test: Union[pd.Series, np.ndarray, str, None] = None,
            dataset_metadata: Dict[str, Any] = frozenset(),
            column_descriptions: Dict[str, str] = None,
            highR_nan_threshold: float = 0.5,
    ):
        '''
        DataManager is a Dataset manager to store the pattern of dataset.

        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param dataset_metadata:
        :param column_descriptions:
        :param highR_nan_threshold:
        '''
        self.logger = get_logger(self)
        dataset_metadata = dict(dataset_metadata)
        self.highR_nan_threshold = highR_nan_threshold
        self.dataset_metadata = dataset_metadata
        X_train = deepcopy(X_train)
        y_train = deepcopy(y_train)
        X_test = deepcopy(X_test)
        y_test = deepcopy(y_test)
        X_train, y_train, X_test, y_test, feature_groups, column2feature_groups = self.parse_column_descriptions(
            column_descriptions, X_train, y_train, X_test, y_test
        )
        self.feature_groups = feature_groups
        self.column2feature_groups = column2feature_groups
        self.ml_task: MLTask = get_ml_task_from_y(y_train)
        self.X_train = GenericDataFrame(X_train, feature_groups=feature_groups)
        self.y_train = y_train
        self.X_test = GenericDataFrame(X_test, feature_groups=feature_groups) if X_test is not None else None
        self.y_test = y_test if y_test is not None else None

        # todo: 用户自定义验证集可以通过RandomShuffle 或者mlxtend指定
        # fixme: 不支持multilabel
        if len(y_train.shape) > 2:
            raise ValueError('y must not have more than two dimensions, '
                             'but has %d.' % len(y_train.shape))

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('X and y must have the same number of '
                             'datapoints, but have %d and %d.' % (X_train.shape[0],
                                                                  y_train.shape[0]))

    def parse_feature_groups(self, series: pd.Series):
        if is_nan(series):
            if is_highR_nan(series, self.highR_nan_threshold):
                return "highR_nan"
            else:
                return "nan"
        elif is_cat(series):
            return "cat"
        else:
            return "num"

    def type_check(self, X):
        if isinstance(X, GenericDataFrame):
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            pass
        else:
            raise TypeError
        return X

    def parse_column_descriptions(self, column_descriptions, X_train, y_train, X_test, y_test):
        # todo: 校验X是否存在重名列
        X_train = self.type_check(X_train)
        X_test = self.type_check(X_test)
        both_set = False
        if X_train is not None and X_test is None:
            X = X_train
            y = y_train
        elif X_train is None and X_test is not None:
            X = X_test
            y = y_test
        elif X_train is not None and X_test is not None:
            both_set = True
            X = X_train
            y = y_train
            self.logger.info("X_train and X_test are both set.")
        else:
            self.logger.error("X_train and X_test are both None, it is invalide.")
            raise ValueError
        if column_descriptions is None:
            column_descriptions = {}
            # fixme : DataManager存在只托管X的情况
            # assert y is not None
        # --确定target--
        if isinstance(y, str) or "target" in column_descriptions:
            if isinstance(y, str):
                target_col = y
            elif "target" in column_descriptions:
                target_col = column_descriptions["target"]
            else:
                raise NotImplementedError
            y_train = pop_if_exists(X_train, target_col)
            y_test = pop_if_exists(X_test, target_col)
        # --确定id--
        if "id" in column_descriptions:
            id_col = column_descriptions["id"]
            self.id_seq = pop_if_exists(X_train, id_col)
            self.test_id_seq = pop_if_exists(X_test, id_col)
        # --确定ignore--
        if "ignore" in column_descriptions:
            ignore_cols = column_descriptions["ignore"]
            if not isinstance(ignore_cols, Sequence):
                ignore_cols = [ignore_cols]
            for ignore_col in ignore_cols:
                pop_if_exists(X_train, ignore_col)
                pop_if_exists(X_test, ignore_col)
        # --验证X与X_test的列应该相同
        if both_set:
            assert X_train.shape[1] == X_test.shape[1]
            assert np.all(X_train.columns == X_test.columns)
        # --确定其他列--
        column2feature_groups = {}
        for key, values in column_descriptions.items():
            if key in ("id", "target", "ignore"):
                continue
            if isinstance(values, str):
                values = [values]
            for value in values:
                column2feature_groups[value] = key
        # ----尝试将X_train与X_test拼在一起，然后做解析---------
        X = stack_Xs(X_train, None, X_test)
        # ----对于没有标注的列，打上nan,highR_nan,cat,num三种标记
        for column in X.columns:
            if column not in column2feature_groups:
                feature_group = self.parse_feature_groups(X[column])
                column2feature_groups[column] = feature_group
        feature_groups = [column2feature_groups[column] for column in X.columns]
        L1 = X_train.shape[0] if X_train is not None else 0
        if X_test is not None:
            L2 = X_test.shape[0]
            X_test.index = range(L1, L1 + L2)
        X_train.index = range(L1)
        return X_train, y_train, X_test, y_test, feature_groups, column2feature_groups

    # def load_from_path(self,X:str):
    #     self.logger.info(f"'{X}' will be parsing as a csv path to load data into data_manager.")
    #     if not self.resource_manager.file_system.exists(X):
    #         self.logger.error(f"'{X}' don't exist in file system.")
    #         raise FileNotFoundError
    #     return self.resource_manager.file_system.load_csv(X)

    def process_X(self, X):
        if X is None:
            return None
        X: pd.DataFrame = self.type_check(X)
        # delete id, ignore, target
        columns = [column for column in X.columns if column in self.column2feature_groups]
        if len(self.feature_groups) != len(columns):
            self.logger.error(
                "In DataManager.process_X, processed columns' length don't equal to feature_groups' length.")
            raise ValueError
        X = X[columns]
        X = GenericDataFrame(X, feature_groups=self.feature_groups)
        return X

    def set_data(self, X_train=None, y_train=None, X_test=None, y_test=None):
        self.X_train = self.process_X(X_train)
        self.X_test = self.process_X(X_test)
        self.y_train = y_train
        self.y_test = y_test

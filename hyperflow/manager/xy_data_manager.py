# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd

from hyperflow.constants import Task
from hyperflow.manager.abstract_data_manager import AbstractDataManager
from hyperflow.pipeline.dataframe import GenericDataFrame
from hyperflow.utils.data import get_task_from_y, is_nan, is_cat
from hyperflow.utils.dataframe import pop_if_exists


class XYDataManager(AbstractDataManager):

    def parse_feat_grp(self, series: pd.Series):
        if is_nan(series):
            return "nan"
        elif is_cat(series):
            return "cat"
        else:
            return "num"

    def parse_column_descriptions(self, column_descriptions, X, y, X_test, y_test):
        # todo: X 为 GeneralDataFrame 的情况
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)
        if column_descriptions is None:
            column_descriptions = {}
            assert y is not None
        # --确定target--
        if isinstance(y, str) or "target" in column_descriptions:
            if isinstance(y, str):
                target_col = y
            elif "target" in column_descriptions:
                target_col = column_descriptions["target"]
            else:
                raise NotImplementedError
            y = X.pop(target_col).values
            y_test = pop_if_exists(X_test, target_col)
        # --确定id--
        if "id" in column_descriptions:
            id_col = column_descriptions["id"]
            self.id_seq = pop_if_exists(X, id_col)
            self.test_id_seq = pop_if_exists(X_test, id_col)
        # --确定ignore--
        if "ignore" in column_descriptions:
            ignore_cols = column_descriptions["ignore"]
            if isinstance(ignore_cols, list):
                ignore_cols = [ignore_cols]

            for ignore_col in ignore_cols:
                pop_if_exists(X, ignore_col)
                pop_if_exists(X_test, ignore_col)
        # --验证X与X_test的列应该相同
        if X_test is not None:
            assert np.all(X.columns == X_test.columns)
        # --确定其他列--
        column2featGrp = {}
        for key, values in column_descriptions.items():
            if key in ("id", "target", "ignore"):
                continue
            if isinstance(values, str):
                values = [values]
            for value in values:
                column2featGrp[value] = key

        # ----对于没有标注的列，打上nan,cat,num三种标记
        for column in X.columns:
            if column not in column2featGrp:
                feat_grp = self.parse_feat_grp(X[column])
                column2featGrp[column] = feat_grp
        feat_grp = [column2featGrp[column] for column in X.columns]
        L1 = X.shape[0]
        if X_test is not None:
            L2 = X_test.shape[0]
            X_test.index = range(L1, L1 + L2)
        X.index = range(L1)
        return X, y, X_test, y_test, feat_grp

    def __init__(
            self, X, y, X_test, y_test, dataset_metadata, column_descriptions
    ):
        super(XYDataManager, self).__init__(dataset_metadata.get("dataset_name", "default_dataset_name"))
        X, y, X_test, y_test, feat_grp = self.parse_column_descriptions(column_descriptions, X, y, X_test, y_test)
        self.task: Task = get_task_from_y(y)
        self.X_train = GenericDataFrame(X, feat_grp=feat_grp)
        self.y_train = y
        self.X_test = GenericDataFrame(X_test, feat_grp=feat_grp) if X_test is not None else None
        self.y_test = y_test if y_test is not None else None

        # todo: 用户自定义验证集可以通过RandomShuffle 或者mlxtend指定
        # fixme: 不支持multilabel
        if len(y.shape) > 2:
            raise ValueError('y must not have more than two dimensions, '
                             'but has %d.' % len(y.shape))

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of '
                             'datapoints, but have %d and %d.' % (X.shape[0],
                                                                  y.shape[0]))

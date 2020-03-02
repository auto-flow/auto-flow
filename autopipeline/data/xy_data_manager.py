# -*- encoding: utf-8 -*-

import numpy as np

from autopipeline.constants import Task, binary_classification_task, multiclass_classification_task, \
    multilabel_classification_task, regression_task
from autopipeline.data.abstract_data_manager import AbstractDataManager
from autopipeline.utils.data import get_task_from_y


class XYDataManager(AbstractDataManager):
    def parse_feature_groups(self, feature_groups):
        # feature_group:
        #    auto: 自动搜索 numerical categorical
        #    list
        if isinstance(feature_groups, str):
            if feature_groups == "auto":
                pass
                # todo  实现
        elif isinstance(feature_groups, list):
            self.feature_groups = feature_groups
        else:
            self.feature_groups = None
        # ----
        if self.feature_groups:
            assert len(self.feature_groups) == self.data['X_train'].shape[1]
        # ----
        if self.feature_groups:
            self.unique_feature_groups = set(self.feature_groups)
        else:
            self.unique_feature_groups = None

    def __init__(
            self, X, y, X_test, y_test, dataset_name, feature_groups
    ):
        super(XYDataManager, self).__init__(dataset_name)
        self.task: Task = get_task_from_y(y)
        self.feature_groups = None
        self.info['has_missing'] = np.all(np.isfinite(X))

        label_num = {
            regression_task: 1,
            binary_classification_task: 2,
            multiclass_classification_task: len(np.unique(y)),
            multilabel_classification_task: y.shape[-1]
        }

        self.info['label_num'] = label_num[self.task]
        # todo: valid
        self.data['X_train'] = X
        self.data['y_train'] = y
        if X_test is not None:
            self.data['X_test'] = X_test
        if y_test is not None:
            self.data['y_test'] = y_test

        # todo: 用户自定义验证集可以通过RandomShuffle 或者mlxtend指定
        # todo: 在kaggle中，需要将 X_test 和 X_train 合并进行 one hot encode 计算
        # todo: 如果用户有y_test ，可能想看到在y_test上的表现

        self.parse_feature_groups(feature_groups)
        # TODO: try to guess task type!
        # fixme: 不支持multilabel
        if len(y.shape) > 2:
            raise ValueError('y must not have more than two dimensions, '
                             'but has %d.' % len(y.shape))

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of '
                             'datapoints, but have %d and %d.' % (X.shape[0],
                                                                  y.shape[0]))


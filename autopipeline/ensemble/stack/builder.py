from copy import deepcopy
from typing import List, Union, Dict

import numpy as np
import pandas as pd
from joblib import load
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression, Lasso

from autopipeline.data.xy_data_manager import XYDataManager
from autopipeline.ensemble.stack.classifier import StackingClassifier
from autopipeline.utils.resource_manager import ResourceManager
from general_fs import LocalFS


def get_model_path_list_of_df(models: DataFrame, df: DataFrame):
    assert hasattr(models, "trial_id")
    trail_ids = models.trial_id
    model_path_list = list(df[df["trail_id"].isin(trail_ids)]["path"])
    return model_path_list


class StackEnsembleBuilder():
    def __init__(
            self,
            set_model: Union[int, List, DataFrame, str] = 10,
            meta_learner=None,
            stack_estimator_kwargs=None
    ):
        self.set_model = set_model
        self.meta_learner = meta_learner
        if not stack_estimator_kwargs:
            stack_estimator_kwargs = {}
        self.stack_estimator_kwargs = stack_estimator_kwargs


    def set_data(
            self,
            data_manager: Union[XYDataManager, Dict],
            dataset_paths: Union[List, str],
            resource_manager: ResourceManager
    ):
        self.dataset_paths = dataset_paths
        self.data_manager = data_manager
        self.resource_manager = resource_manager
        self.file_system=resource_manager.file_system

    def init_data(self):
        if not self.file_system:
            self.file_system = LocalFS()
        self.task = self.data_manager.task
        if not self.meta_learner:
            if self.task.mainTask == "classification":
                self.meta_learner = LogisticRegression(penalty='l2', solver="lbfgs", multi_class="auto",
                                                       random_state=10)
            else:
                self.meta_learner = Lasso()
        self.set_model = self.set_model
        if isinstance(self.dataset_paths, str):
            self.dataset_paths = [self.dataset_paths]

    def build(self):
        set_model = self.set_model

        if isinstance(set_model, int):
            trial_ids=self.resource_manager.get_best_k_trials(set_model)
        elif isinstance(set_model, list):
            trial_ids=deepcopy(set_model)
            # todo: 验证
        elif isinstance(set_model, DataFrame):
            raise NotImplementedError
        elif isinstance(set_model, str):
            raise NotImplementedError
        else:
            raise NotImplementedError()
        estimator_list, y_true_indexes, y_preds_list=\
            self.resource_manager.load_estimators_in_trials(trial_ids)
        if self.task.mainTask == "classification":
            stack_estimator_cls = StackingClassifier
        else:
            raise NotImplementedError()

        stack_estimator = stack_estimator_cls(
            self.meta_learner,
            estimator_list, y_true_indexes, y_preds_list,
            **self.stack_estimator_kwargs
        )
        stack_estimator.fit(self.data_manager.data["X_train"], self.data_manager.data["y_train"])
        return stack_estimator

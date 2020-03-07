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
        df_list = []
        for dataset_path in self.dataset_paths:
            df = pd.read_csv(dataset_path + "/trials.csv")
            df["dir"] = [dataset_path + "/trials"] * df.shape[0]
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        df["path"] = df["dir"].str.cat(df["trial_id"], sep="/") + ".bz2"
        df.sort_values(by=["loss", "cost_time"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        if isinstance(set_model, int):
            model_path_list = list(df.loc[:set_model - 1, "path"])
        elif isinstance(set_model, list):
            assert all(map(self.file_system.exists, set_model))
            model_path_list = set_model
        elif isinstance(set_model, DataFrame):
            model_path_list = get_model_path_list_of_df(set_model, df)
        elif isinstance(set_model, str):
            set_model = pd.read_csv(set_model)
            model_path_list = get_model_path_list_of_df(set_model, df)
        else:
            raise NotImplementedError()
        models_list = []
        prediction_list = []
        for model_path in model_path_list:
            data = load(model_path)
            y_test_indices = data["y_test_indices"]
            y_preds = data["y_preds"]
            models = data["models"]
            prediction = np.zeros_like(np.vstack(y_preds))
            for y_index, y_pred in zip(y_test_indices, y_preds):
                prediction[y_index] = y_pred
            prediction_list.append(prediction)
            models_list.append(models)
            # todo: 内存溢出？
        if self.task.mainTask == "classification":
            stack_estimator_cls = StackingClassifier
        else:
            raise NotImplementedError()

        stack_estimator = stack_estimator_cls(
            self.meta_learner,
            models_list,
            prediction_list,
            **self.stack_estimator_kwargs
        )
        stack_estimator.model_path_list = model_path_list
        stack_estimator.fit(self.data_manager.data["X_train"], self.data_manager.data["y_train"])
        return stack_estimator

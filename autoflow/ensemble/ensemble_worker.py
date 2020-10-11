#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from multiprocessing import Process
from time import sleep
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import check_random_state

from autoflow.data_manager import DataManager
from autoflow.evaluation.train_evaluator import TrainEvaluator
from autoflow.hdl.hdl2cs import HDL2CS
from autoflow.hdl.hdl_constructor import HDL_Constructor
from autoflow.utils.ml_task import MLTask


class EnsembleWorker(Process):

    def __init__(self, resource_manager, task_id, budget_id, metric, ml_task: MLTask, random_state=0):
        super(EnsembleWorker, self).__init__()
        self.ml_task = ml_task
        self.metric = metric
        self.budget_id = budget_id
        self.task_id = task_id
        self.resource_manager = resource_manager
        self.y_train = None
        self.y_test = None
        self.have_X_test = None
        self.trial2y_info = {}
        self.rng = check_random_state(random_state)

    def fetch_meta_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        X_meta_train, X_meta_test = [], []
        columns = []
        n_columns_per_model = 0
        for i, row in df.iterrows():
            trial_id = row["trial_id"]
            y_info_path = row["y_info_path"]
            if trial_id in self.trial2y_info:
                y_info = self.trial2y_info[trial_id]
            else:
                y_info = self.resource_manager.file_system.load_pickle(y_info_path)
                self.trial2y_info[trial_id] = y_info
            y_true_indexes = y_info['y_true_indexes']
            y_test_pred = y_info.get('y_test_pred')
            y_preds = y_info['y_preds']
            y_train_pred = np.full([self.y_train.shape[0], y_preds[0].shape[1]], np.nan)
            if self.have_X_test is None:
                if y_test_pred is None:
                    self.have_X_test = False
                else:
                    self.have_X_test = True
            for y_pred, y_true_indexes in zip(y_preds, y_true_indexes):
                y_train_pred[y_true_indexes] = y_pred
            L = y_train_pred.shape[1]
            if self.ml_task.mainTask == "regression":
                y_train_pred = y_train_pred.reshape(-1, 1)
                y_test_pred = y_test_pred.reshape(-1, 1) if y_test_pred is not None else None
            if self.ml_task.mainTask == "classification":
                y_train_pred = y_train_pred[:, 1:]
                y_test_pred = y_test_pred[:, 1:] if y_test_pred is not None else None
                L -= 1
            n_columns_per_model = L
            if L > 1:
                columns.extend([f"{trial_id}_{j}" for j in range(L)])
            else:
                columns += [str(trial_id)]
            X_meta_train.append(y_train_pred)
            if self.have_X_test:
                X_meta_test.append(y_test_pred)
        X_meta_train = np.hstack(X_meta_train)
        X_meta_train = pd.DataFrame(X_meta_train, columns=columns)
        X_meta_test = pd.DataFrame(np.hstack(X_meta_test), columns=columns) if self.have_X_test else None
        return X_meta_train, X_meta_test, n_columns_per_model

    def compute(self):
        # 如果y_true不存在，获取之
        if self.y_train is None:
            from autoflow.data_container import NdArrayContainer

            # 操作task而不是trial
            self.resource_manager.init_task_table()
            task_records = self.resource_manager._get_task_records(self.task_id, self.resource_manager.user_id)
            assert len(task_records) > 0
            task_record = task_records[0]
            ml_task_dict = task_record["ml_task"]
            self.ml_task = MLTask(**ml_task_dict)
            train_label_id = task_record["train_label_id"]
            test_label_id = task_record["test_label_id"]
            y_train = NdArrayContainer(dataset_id=train_label_id, resource_manager=self.resource_manager)
            if test_label_id:
                y_test = NdArrayContainer(dataset_id=test_label_id, resource_manager=self.resource_manager)
                self.y_test = y_test.data
            else:
                self.have_X_test = False
            self.y_train = y_train.data
        # 查询记录
        rm.init_trial_table()
        Trial = rm.TrialModel
        records = list(Trial.select(
            Trial.trial_id,
            Trial.loss, Trial.test_loss,
            Trial.additional_info["best_iterations"].alias("best_iterations"), Trial.status, Trial.y_info_path
        ).where(
            (Trial.task_id == self.task_id) & (Trial.budget_id == self.budget_id)
        ).dicts())
        df = pd.DataFrame(records, columns=["trial_id", "loss", "test_loss", "best_iterations", "status",
                                            "y_info_path"])  # todo： 考虑再取出 score ？
        # todo: 考虑集成模型数的最大值与最小值
        # 按loss取前20%
        quantile = np.quantile(df["loss"], 0.5)
        df = df.query(f"loss <= {quantile}").sort_values(by="loss")
        df.index = range(df.shape[0])
        # 根据筛选出来的记录构造元特征（训练集与测试集）
        X_meta_train, X_meta_test, n_columns_per_model = self.fetch_meta_features(df)
        assert X_meta_train.shape[1] == df.shape[0] * n_columns_per_model, ValueError
        # todo：提供一个option，可以抛弃没用完成所有CV（即meta_features存在nan）的列
        groups = []
        for i in range(X_meta_train.shape[1]):
            groups += [i] * n_columns_per_model
        groups = np.array(groups)
        groups_unq = np.arange(X_meta_train.shape[1])
        X_meta_train_imputed = X_meta_train.copy().values
        nan_group = np.unique(groups[X_meta_train.isna().sum() > 0])
        full_group = np.setdiff1d(groups_unq, nan_group)
        target_group_for_nan = []
        for group_id in nan_group:
            loss = df.loc[group_id, "loss"]
            other_df = df.loc[full_group]
            idx = np.argmin(np.abs(other_df["loss"] - loss))
            target_group_id = other_df.iloc[idx].name
            target_group_for_nan.append(target_group_id)
            mask = X_meta_train.loc[:, groups == group_id].isna()
            nan_rows = np.arange(X_meta_train.shape[0])[mask.sum(axis=1) > 0]
            # todo: 加一点随机噪音？
            target_value = X_meta_train.values[nan_rows, groups == target_group_id]
            target_value += (self.rng.randn(*target_value.shape) * 0.02)
            X_meta_train_imputed[nan_rows, groups == group_id] = target_value
        X_meta_train = pd.DataFrame(X_meta_train_imputed, columns=X_meta_train.columns)
        X_meta_train_ = X_meta_train.copy()
        X_meta_train_["label"] = self.y_train
        if self.have_X_test:
            X_meta_test_ = X_meta_test.copy()
            X_meta_test_["label"] = self.y_test
        else:
            X_meta_test_ = None
        # todo: 将元特征存储在experiment对应的文件夹中，必要时取出来调试
        X_meta_train_.to_csv("X_meta_train.csv", index=False)
        if X_meta_test_ is not None:
            X_meta_test_.to_csv("X_meta_test.csv", index=False)
        data_manager = DataManager(self.resource_manager, X_meta_train, self.y_train, X_meta_test, self.y_test)
        DAG_workflow = {
            "num->selected": {
                "_name": "select.boruta",
                "max_depth": 7,
                "weak": False
                # todo: groups
            },
            # LR, SVM, KNN
            # todo: 针对不同的数据量，对算法进行评测
            "selected->target": [
                {
                    "_name": "linearsvc"
                },
                {
                    "_name": "logistic_regression",

                },
                {
                    "_name": "tabular_nn"
                }
            ],
        }

        hdl_constructor = HDL_Constructor(DAG_workflow)
        hdl_constructor.run(data_manager)
        hdl = hdl_constructor.get_hdl()
        cs = HDL2CS()(hdl)
        # todo: 是否要做交叉验证？
        if self.ml_task.mainTask == "classification":
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        else:
            splitter = KFold(n_splits=5, shuffle=True, random_state=0)
        train_evaluator = TrainEvaluator(
            "", data_manager, self.resource_manager, 0, self.metric, None, True, splitter, \
            insert_trial_table=False
            # todo: refit
        )
        res = train_evaluator.compute(cs.sample_configuration().get_dictionary(), {}, 1)
        # train_evaluator()
        print(records)
        # 用rm获取

    def run(self) -> None:
        while True:
            self.compute()
            sleep(1)


if __name__ == '__main__':
    from autoflow.resource_manager.base import ResourceManager
    from autoflow.metrics import accuracy
    from autoflow.constants import binary_classification_task

    db_params = {
        "user": "tqc",
        "host": "127.0.0.1",
        "port": 5432,
    }
    search_record_db_name = "autoflow_meta_bo"
    rm = ResourceManager(
        db_params=db_params,
        db_type="postgresql",
        store_path=f"~/{search_record_db_name}",
        search_record_db_name=search_record_db_name
    )
    rm.experiment_id = 0
    rm.init_trial_table()
    print("start")
    p = EnsembleWorker(rm, "439f1de1aa95d757d005c9c5c60e3b63", "fa059a2f2be19553029628cbeb1c59ad", accuracy,
                       binary_classification_task)
    p.compute()

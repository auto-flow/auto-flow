#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
import pandas as pd
from pandas import Index

from autoflow import DataManager
from autoflow import datasets
from autoflow.tests.base import LocalResourceTestCase
from autoflow.utils.dict_ import sort_dict


class TestDataManager(LocalResourceTestCase):
    def test_instancing1(self):
        def do_assert(data_manager, remote=False, stacked=True):
            final_column_descriptions = {'id': 'PassengerId',
                                         'target': 'Survived',
                                         'text': ['Name'],
                                         'num': ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
                                         'cat': ['Sex', 'Cabin', 'Embarked'],
                                         'highC_cat': ['Ticket']}
            assert sort_dict(data_manager.final_column_descriptions) == sort_dict(final_column_descriptions)
            if not remote:
                assert sort_dict(data_manager.column_descriptions) == sort_dict({'id': 'PassengerId', 'target': 'Survived', 'text': 'Name'})
            else:
                assert sort_dict(data_manager.column_descriptions) == sort_dict(final_column_descriptions)
            if stacked:
                assert np.all(pd.Series(data_manager.feature_groups) == pd.Series(['num', 'text', 'cat', 'nan',
                                                                                   'num', 'num', 'highC_cat', 'nan',
                                                                                   'highR_nan', 'nan']))
            else:
                assert np.all(pd.Series(data_manager.feature_groups) == pd.Series(['num', 'text', 'cat', 'nan',
                                                                                   'num', 'num', 'highC_cat', 'num',
                                                                                   'highR_nan', 'nan']))
            assert np.all(
                data_manager.columns == Index(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
                                               'Cabin', 'Embarked'],
                                              dtype='object'))

        train_df, test_df = datasets.load("titanic", return_train_test=True)
        column_descriptions = {
            "id": "PassengerId",
            "target": "Survived",
            "text": "Name"
        }
        data_manager1 = DataManager(
            X_train=train_df,
            X_test=test_df,
            column_descriptions=column_descriptions,
            resource_manager=self.mock_resource_manager
        )
        do_assert(data_manager1, remote=False, stacked=True)
        # -------------------------------------------------------------------------------

        train_df, test_df = datasets.load("titanic", return_train_test=True)
        data_manager2 = DataManager(
            X_train=train_df,
            # X_test=test_df,
            column_descriptions=column_descriptions,
            resource_manager=self.mock_resource_manager
        )
        do_assert(data_manager2, remote=False, stacked=False)
        # -------------------------------------------------------------------------------

        data_manager3 = DataManager(
            X_train=data_manager1.train_set_id,
            # X_test=test_df,
            column_descriptions=column_descriptions,
            resource_manager=self.mock_resource_manager
        )
        do_assert(data_manager3, remote=True, stacked=False)
        # -------------------------------------------------------------------------------

        data_manager4 = DataManager(
            X_train=data_manager1.train_set_id,
            X_test=data_manager1.test_set_id,
            column_descriptions=column_descriptions,
            resource_manager=self.mock_resource_manager
        )
        do_assert(data_manager4, remote=True, stacked=True)

    def test_reindex_columns(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        from autoflow.core.classifier import AutoFlowClassifier

        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        pipe = AutoFlowClassifier(
            DAG_workflow={
                "num->target": [
                    "logistic_regression",
                ]
            },
            initial_runs=1,
            run_limit=1,
            n_jobs=1,
            debug=True,
            search_method="smac",
            random_state=0,
            resource_manager=self.mock_resource_manager
        )
        pipe.fit(X_train, y_train, X_test, y_test)
        X_test = pipe.data_manager.X_test.data
        X_test = X_test[[f'column_{i}' for i in range(3, -1, -1)]]
        # score = accuracy_score(y_test, y_pred)
        score = pipe.score(X_test, y_test)
        print(score)
        assert score > 0.8
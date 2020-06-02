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


class TestDataManager(LocalResourceTestCase):
    def test_instancing1(self):
        def do_assert(data_manager, remote=False, stacked=True):
            final_column_descriptions = {'id': 'PassengerId',
                                         'target': 'Survived',
                                         'text': ['Name'],
                                         'num': ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
                                         'cat': ['Sex', 'Cabin', 'Embarked'],
                                         'highR_cat': ['Ticket']}
            assert data_manager.final_column_descriptions == final_column_descriptions
            if not remote:
                assert data_manager.column_descriptions == {'id': 'PassengerId', 'target': 'Survived', 'text': 'Name'}
            else:
                assert data_manager.column_descriptions == final_column_descriptions
            if stacked:
                assert np.all(pd.Series(data_manager.feature_groups) == pd.Series(['num', 'text', 'cat', 'nan',
                                                                                   'num', 'num', 'highR_cat', 'nan',
                                                                                   'highR_nan', 'nan']))
            else:
                assert np.all(pd.Series(data_manager.feature_groups) == pd.Series(['num', 'text', 'cat', 'nan',
                                                                                   'num', 'num', 'highR_cat', 'num',
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
            column_descriptions=column_descriptions
        )
        do_assert(data_manager1, remote=False, stacked=True)
        # -------------------------------------------------------------------------------

        train_df, test_df = datasets.load("titanic", return_train_test=True)
        data_manager2 = DataManager(
            X_train=train_df,
            # X_test=test_df,
            column_descriptions=column_descriptions
        )
        do_assert(data_manager2, remote=False, stacked=False)
        # -------------------------------------------------------------------------------

        data_manager3 = DataManager(
            X_train=data_manager1.train_set_hash,
            # X_test=test_df,
            column_descriptions=column_descriptions
        )
        do_assert(data_manager3, remote=True, stacked=False)
        # -------------------------------------------------------------------------------

        data_manager4 = DataManager(
            X_train=data_manager1.train_set_hash,
            X_test=data_manager1.test_set_hash,
            column_descriptions=column_descriptions
        )
        do_assert(data_manager4, remote=True, stacked=True)

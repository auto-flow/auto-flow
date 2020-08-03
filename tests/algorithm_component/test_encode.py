#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from time import time

import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

from autoflow import AutoFlowRegressor
from autoflow.data_container import DataFrameContainer, NdArrayContainer
from autoflow.hdl.utils import get_default_hp_of_cls
from autoflow.tests.base import LocalResourceTestCase
from autoflow.workflow.components.preprocessing.encode.binary import BinaryEncoder
from autoflow.workflow.components.preprocessing.encode.cat_boost import CatBoostEncoder
from autoflow.workflow.components.preprocessing.encode.entity import EntityEncoder
from autoflow.workflow.components.preprocessing.encode.leave_one_out import LeaveOneOutEncoder
from autoflow.workflow.components.preprocessing.encode.one_hot import OneHotEncoder
from autoflow.workflow.components.preprocessing.encode.ordinal import OrdinalEncoder
from autoflow.workflow.components.preprocessing.encode.target import TargetEncoder
from autoflow.workflow.components.regression.random_forest import RandomForestRegressor
from autoflow.workflow.ml_workflow import ML_Workflow


class TestCoding(LocalResourceTestCase):
    def setUp(self) -> None:
        super(TestCoding, self).setUp()
        boston = load_boston()
        np.random.seed(10)
        ix = np.random.permutation(boston.data.shape[0])
        L = int(len(ix) * 0.8)
        train_ix = ix[:L]
        test_ix = ix[L:]
        X_train = pd.DataFrame(boston.data[train_ix, :], columns=boston.feature_names)
        y_train = boston.target[train_ix]
        X_test = pd.DataFrame(boston.data[test_ix, :], columns=boston.feature_names)
        y_test = boston.target[test_ix]
        pipe = AutoFlowRegressor(
            consider_ordinal_as_cat=True,
            resource_manager=self.mock_resource_manager
        )
        pipe.fit(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            is_not_realy_run=True,

        )
        self.X_train = pipe.data_manager.X_train
        self.X_train.index = train_ix
        self.y_train = pipe.data_manager.y_train
        self.X_test = pipe.data_manager.X_test
        self.X_test.index = test_ix
        self.y_test = pipe.data_manager.y_test
        self.index = deepcopy(train_ix)

    def test_io(self):
        for cls in [
            TargetEncoder,
            BinaryEncoder,
            CatBoostEncoder,
            OrdinalEncoder,
            LeaveOneOutEncoder,
            OneHotEncoder,
            # WOEEncoder,  # 不支持回归
        ]:
            encoder = cls(
                in_feature_groups="cat",
                out_feature_groups="encoded"
            )
            trans = encoder.fit_transform(X_train=self.X_train, X_valid=self.X_test, y_train=self.y_train)["X_train"]
            encoded = trans.filter_feature_groups("encoded")
            assert np.all(encoded.index == self.index)
            assert np.all(trans.filter_feature_groups("num").data == self.X_train.filter_feature_groups("num").data)
            pre_columns = self.X_train.filter_feature_groups("cat").columns
            for column in encoded.columns:
                ok = False
                for pre_column in pre_columns:
                    if column.startswith(pre_column):
                        ok = True
                        break
                assert ok

    def test_procedure(self):
        for cls in [
            EntityEncoder,
            TargetEncoder,
            BinaryEncoder,
            CatBoostEncoder,
            OrdinalEncoder,
            LeaveOneOutEncoder,
            OneHotEncoder,
            # WOEEncoder,  # 不支持回归
        ]:
            print("=========================")
            print(cls.__name__)
            print("=========================")
            start = time()
            hp = get_default_hp_of_cls(cls)
            workflow = ML_Workflow(steps=[
                ("encoder", cls(
                    in_feature_groups="cat",
                    out_feature_groups="num",
                    **hp
                )),
                ("rf", RandomForestRegressor(
                    random_state=0
                ))
            ], resource_manager=self.mock_resource_manager)
            workflow.fit(X_train=self.X_train, X_valid=self.X_test, y_train=self.y_train, y_valid=self.y_test)
            y_pred = workflow.predict(self.X_test)
            score = r2_score(self.y_test.data, y_pred)
            print("r2 = ", score)
            print("time = ", time() - start)
            print("\n" * 2)

    def test_ordinal_encode_category(self):
        df2 = pd.DataFrame([
            ['C', '3'],
            ['D', '4'],
            ['D', '4'],
        ], columns=['alpha', 'digits'])
        df2["digits"] = df2["digits"].astype(CategoricalDtype(categories=["4", "3"], ordered=True))
        df2["alpha"] = df2["alpha"].astype(CategoricalDtype(categories=["D", "C"], ordered=True))
        df2_ = df2.loc[1:, :]
        df2_1 = df2.loc[:1, :]
        df2_c = pd.concat([df2_, df2_1])
        df2_c.index = range(4)
        encoder = OrdinalEncoder()

        encoder.in_feature_groups = "cat"
        encoder.out_feature_groups = "ordinal"
        # RunFeatureSelection().test_univar_clf()
        # RunCoding().test_procedure()
        dc = DataFrameContainer(dataset_instance=df2_c)
        dc.set_feature_groups(["cat"] * 2)
        encoder.fit(X_train=dc)
        result = encoder.transform(X_train=dc)["X_train"]
        print(result)
        should_be = pd.DataFrame({'alpha': {0: 0, 1: 0, 2: 1, 3: 0}, 'digits': {0: 0, 1: 0, 2: 1, 3: 0}})
        assert np.all(result.data == should_be)

    def test_handle_unknown(self):
        X_train = pd.DataFrame([
            ['A', 'alpha', 9],
            ['A', 'alpha', 1],
            ['B', 'beta', 2],
            ['B', 'beta', 3],
            ['C', 'gamma', 4],
            ['C', 'gamma', 5],
        ], columns=['col1', 'col2', 'col3'])
        X_valid = pd.DataFrame([
            ['D', 'kappa', 6],
            ['D', 'kappa', 6],
            ['E', 'sigma', 7],
            ['E', 'sigma', 7],
            ['F', 'mu', 8],
            ['F', 'mu', 8],
        ], columns=['col1', 'col2', 'col3'])
        X_train = DataFrameContainer(dataset_instance=X_train)
        X_valid = DataFrameContainer(dataset_instance=X_valid)
        X_train.set_feature_groups(['cat'] * 3)
        X_valid.set_feature_groups(['cat'] * 3)
        y_train = NdArrayContainer(dataset_instance=[0, 1, 0, 1, 0, 1])
        for cls in [EntityEncoder, OrdinalEncoder, OneHotEncoder, TargetEncoder, CatBoostEncoder]:
            hp = get_default_hp_of_cls(cls)
            encoder = cls(**hp)
            encoder.in_feature_groups = "cat"
            encoder.out_feature_groups = "ordinal"
            result = encoder.fit_transform(X_train=X_train, X_valid=X_valid, y_train=y_train)
            assert np.all(encoder.transform(X_train)['X_train'].data == result['X_train'].data)
            assert np.all(encoder.transform(X_valid)['X_train'].data == result['X_valid'].data)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from time import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

from autoflow import AutoFlowRegressor
from autoflow.hdl.utils import get_default_hp_of_cls
from autoflow.tests.base import LocalResourceTestCase
from autoflow.workflow.components.preprocessing.operate.keep_going import KeepGoing
from autoflow.workflow.components.preprocessing.scale.minmax import MinMaxScaler
from autoflow.workflow.components.preprocessing.scale.normalize import Normalizer
from autoflow.workflow.components.preprocessing.scale.quantile import QuantileTransformer
from autoflow.workflow.components.preprocessing.scale.robust import RobustScaler
from autoflow.workflow.components.preprocessing.scale.standardize import StandardScaler
from autoflow.workflow.components.regression.linearsvr import LinearSVR
from autoflow.workflow.ml_workflow import ML_Workflow


class TestScaler(LocalResourceTestCase):
    def setUp(self) -> None:
        super(TestScaler, self).setUp()
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
            # consider_ordinal_as_cat=True,
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
            MinMaxScaler,
            StandardScaler,
            Normalizer,
            QuantileTransformer,
            RobustScaler,
            KeepGoing,
        ]:
            encoder = cls(
                **get_default_hp_of_cls(cls)
            )
            encoder.in_feature_groups = "num"
            encoder.out_feature_groups = "final"
            trans = encoder.fit_transform(X_train=self.X_train, X_valid=self.X_test, y_train=self.y_train)["X_train"]
            assert np.all(trans.feature_groups == "final")
            assert np.all(trans.index == self.index)
            assert np.all(trans.columns == self.X_train.columns)

    def test_procedure(self):
        for cls in [
            MinMaxScaler,
            StandardScaler,
            Normalizer,
            QuantileTransformer,
            RobustScaler,
            KeepGoing,
            # WOEEncoder,  # 不支持回归
        ]:
            print("=========================")
            print(cls.__name__)
            print("=========================")
            if cls == KeepGoing:
                hp = {}
            else:
                hp = get_default_hp_of_cls(cls)
            start = time()
            workflow = ML_Workflow(steps=[
                ("scaler", cls(
                    in_feature_groups="num",
                    out_feature_groups="scaled",
                    **hp
                )),
                ("rf", LinearSVR(
                    random_state=0
                ))
            ], resource_manager=self.mock_resource_manager)
            workflow.fit(X_train=self.X_train, X_valid=self.X_test, y_train=self.y_train, y_valid=self.y_test)
            y_pred = workflow.predict(self.X_test)
            score = r2_score(self.y_test.data, y_pred)
            print("r2 = ", score)
            print("time = ", time() - start)
            print("\n" * 2)
    #

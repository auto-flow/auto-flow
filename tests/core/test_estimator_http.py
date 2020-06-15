#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import subprocess
import sys
import time
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from autoflow import HDL_Constructor, NdArrayContainer
from autoflow.core.classifier import AutoFlowClassifier
from autoflow.ensemble.stack.classifier import StackClassifier
from autoflow.ensemble.vote.classifier import VoteClassifier
from autoflow.resource_manager.http import HttpResourceManager
from autoflow.tests.base import LocalResourceTestCase
from autoflow.tuner import Tuner
from autoflow.datasets import load
from autoflow.data_container.dataframe import DataFrameContainer
import numpy as np

class TestEstimatorsHttp(LocalResourceTestCase):
    def setUp(self) -> None:
        super(TestEstimatorsHttp, self).setUp()
        # todo: 考虑8000端口已经占用的情况
        self.p = subprocess.Popen(
            args=[self.get_uvicorn_path(), "server:app"],
            cwd=self.get_server_dir(),
            env={
                "LC_ALL": "C.UTF-8",
                "LANG": "C.UTF-8",
                "DB_TYPE": "sqlite",
                "STORE_PATH": self.mock_resource_manager.store_path,
            },
        )
        time.sleep(2)
        self.http_mock_resource_manager = HttpResourceManager(
            db_params={
                "http_client": True,
                "url": "http://127.0.0.1:8000",
                "headers": {
                    'Content-Type': 'application/json',
                    'accept': 'application/json',
                }
            }
        )

    def tearDown(self) -> None:
        super(TestEstimatorsHttp, self).tearDown()
        self.p.kill()

    def get_server_dir(self):
        server_dir = (Path(__file__).parent.parent.parent / "autoflow_server").as_posix()
        return server_dir

    def get_uvicorn_path(self):
        return (Path(sys.executable).parent / "uvicorn").as_posix()

    def test_1(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        pipe = AutoFlowClassifier(
            DAG_workflow={
                "num->target": ["liblinear_svc", "libsvm_svc", "logistic_regression"]
            },
            initial_runs=3,
            run_limit=9,
            n_jobs=3,
            debug=True,
            resource_manager=self.http_mock_resource_manager
        )
        pipe.fit(
            X_train, y_train,
            fit_ensemble_params="auto"
        )
        assert isinstance(pipe.estimator, StackClassifier)
        score = pipe.score(X_test, y_test)
        assert score > 0.8

    def test_2(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        hdl_constructors = [
            HDL_Constructor(
                DAG_workflow={
                    "num->target": ["liblinear_svc", "libsvm_svc", "logistic_regression"]
                },
            )
        ]*2
        tuners = [
            Tuner(
                search_method="random",
                run_limit=3,
                n_jobs=3,
                debug=True
            ),
            Tuner(
                search_method="smac",
                initial_runs=3,
                run_limit=6,
                n_jobs=3,
                debug=True
            )
        ]
        pipe = AutoFlowClassifier(
            hdl_constructor=hdl_constructors,
            tuner=tuners,
            resource_manager=self.http_mock_resource_manager
        )
        pipe.fit(
            X_train, y_train,
            fit_ensemble_params=False
        )
        assert isinstance(pipe.estimator, VoteClassifier)
        # score = accuracy_score(y_test, y_pred)
        score = pipe.score(X_test, y_test)
        assert score > 0.8

    def test_3(self):
        titanic_df = load("titanic")
        titanic_df.index = reversed(titanic_df.index)
        dc = DataFrameContainer(dataset_instance=titanic_df, resource_manager=self.http_mock_resource_manager)
        feat_grp = [f"feat_{i}" for i in range(dc.shape[1])]
        dc.set_feature_groups(feat_grp)
        column_descriptions = dc.column_descriptions
        dc.upload()
        dataset_id = dc.dataset_id
        download_dc = DataFrameContainer("Unittest", dataset_id=dataset_id, resource_manager=self.http_mock_resource_manager)
        self.assertTrue(np.all(download_dc.data.fillna(0) == dc.data.fillna(0)))
        self.assertTrue(np.all(download_dc.feature_groups == dc.feature_groups))
        self.assertTrue(np.all(download_dc.columns == dc.columns))
        self.assertTrue(np.all(download_dc.index == dc.index))
        self.assertEqual(download_dc.column_descriptions, dc.column_descriptions)
        self.assertEqual(download_dc.columns_mapper, dc.columns_mapper)
        self.assertEqual(download_dc.dataset_type, dc.dataset_type)
        self.assertEqual(download_dc.dataset_source, dc.dataset_source)
        ###################################################################
        in_data = [1, 2, 3, 4, 5]
        dc = NdArrayContainer(dataset_instance=in_data, resource_manager=self.http_mock_resource_manager)
        dc.upload()
        d_dc = NdArrayContainer(dataset_id=dc.dataset_id, resource_manager=self.http_mock_resource_manager)
        self.assertTrue(np.all(d_dc.data == np.array(in_data)))

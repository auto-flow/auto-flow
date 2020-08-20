#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from collections import Counter
from time import time

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from autoflow import datasets
from autoflow.data_container import DataFrameContainer
from autoflow.data_container import NdArrayContainer
from autoflow.hdl.utils import get_default_hp_of_cls
from autoflow.tests.base import LocalResourceTestCase
from autoflow.workflow.components.classification.linearsvc import LinearSVC
# todo: 调研并增加超参数
from autoflow.workflow.components.preprocessing.balance.under_sample.all_knn import AllKNN
from autoflow.workflow.components.preprocessing.balance.under_sample.cluster_centroids import ClusterCentroids
from autoflow.workflow.components.preprocessing.balance.under_sample.condensed_nearest_neighbour import CondensedNearestNeighbour
from autoflow.workflow.components.preprocessing.balance.under_sample.edited_nearest_neighbours import EditedNearestNeighbours
from autoflow.workflow.components.preprocessing.balance.under_sample.instance_hardness_threshold import InstanceHardnessThreshold
from autoflow.workflow.components.preprocessing.balance.under_sample.near_miss import NearMiss
from autoflow.workflow.components.preprocessing.balance.under_sample.neighbourhood_cleaning_rule import NeighbourhoodCleaningRule
from autoflow.workflow.components.preprocessing.balance.under_sample.one_sided_selection import OneSidedSelection
from autoflow.workflow.components.preprocessing.balance.under_sample.random import RandomUnderSampler
from autoflow.workflow.components.preprocessing.balance.under_sample.repeated_edited_nearest_neighbours import RepeatedEditedNearestNeighbours
from autoflow.workflow.components.preprocessing.balance.under_sample.tomek_links import TomekLinks

from autoflow.workflow.components.preprocessing.balance.over_sample.random import RandomOverSampler
from autoflow.workflow.components.preprocessing.balance.over_sample.adasyn import ADASYN
from autoflow.workflow.components.preprocessing.balance.over_sample.borderline_smote import BorderlineSMOTE
from autoflow.workflow.components.preprocessing.balance.over_sample.kmeans_smote import KMeansSMOTE
from autoflow.workflow.components.preprocessing.balance.over_sample.smote import SMOTE
from autoflow.workflow.components.preprocessing.balance.over_sample.svmsmote import SVMSMOTE
from sklearn.datasets import load_iris

from autoflow.workflow.ml_workflow import ML_Workflow


class TestBalance(LocalResourceTestCase):
    def setUp(self) -> None:
        super(TestBalance, self).setUp()
        X, y=load_iris(return_X_y=True)
        y[y==2]=1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=0)
        X_train = DataFrameContainer("TrainSet", dataset_instance=X_train, resource_manager=self.mock_resource_manager)
        X_test = DataFrameContainer("TestSet", dataset_instance=X_test, resource_manager=self.mock_resource_manager)
        y_train = NdArrayContainer("TrainLabel", dataset_instance=y_train, resource_manager=self.mock_resource_manager)
        y_test = NdArrayContainer("TestLabel", dataset_instance=y_test, resource_manager=self.mock_resource_manager)
        X_train.set_feature_groups(["num"]*4)
        X_test.set_feature_groups(["num"]*4)
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test

    def test_under_sample(self):

        est_cls_list = [
            AllKNN,
            ClusterCentroids,
            CondensedNearestNeighbour,
            EditedNearestNeighbours,
            InstanceHardnessThreshold,
            NearMiss,
            NeighbourhoodCleaningRule,
            OneSidedSelection,
            RandomUnderSampler,
            RepeatedEditedNearestNeighbours,
            TomekLinks,
        ]

        for cls in est_cls_list:
            print("=========================")
            print(cls.__name__)
            print("=========================")
            balancer = cls(
                **get_default_hp_of_cls(cls)
            )
            classifier = LinearSVC(
                **get_default_hp_of_cls(LinearSVC)
            )
            pipeline = ML_Workflow(
                [
                    ("balancer", balancer),
                    ("classifier", classifier),
                ],
                resource_manager=self.mock_resource_manager,
                should_store_intermediate_result=True
            )
            start = time()
            pipeline.fit(self.X_train, self.y_train, self.X_test, self.y_test)
            balanced_y_train = NdArrayContainer(dataset_id=pipeline.intermediate_result["balancer"]["y_train"],
                                       resource_manager=self.mock_resource_manager)
            print("y_train:")
            print(Counter(self.y_train.data))
            print("balanced y_train:")
            print(Counter(balanced_y_train.data))

            y_pred = pipeline.predict(self.X_test)
            score = accuracy_score(self.y_test.data, y_pred)
            end = time()
            print("score:", score)
            print("time:", end - start)
            self.assertGreater(score, 0.6)
            print('\n' * 2)

    def test_over_sample(self):
        est_cls_list = [
            RandomOverSampler,
            # ADASYN,
            BorderlineSMOTE,
            KMeansSMOTE,
            SMOTE,
            SVMSMOTE,
        ]

        for cls in est_cls_list:
            print("=========================")
            print(cls.__name__)
            print("=========================")
            balancer = cls(
                **get_default_hp_of_cls(cls)
            )
            classifier = LinearSVC(
                **get_default_hp_of_cls(LinearSVC)
            )
            pipeline = ML_Workflow(
                [
                    ("balancer", balancer),
                    ("classifier", classifier),
                ],
                resource_manager=self.mock_resource_manager,
                should_store_intermediate_result=True
            )
            start = time()
            pipeline.fit(self.X_train, self.y_train, self.X_test, self.y_test)
            balanced_y_train = NdArrayContainer(dataset_id=pipeline.intermediate_result["balancer"]["y_train"],
                                       resource_manager=self.mock_resource_manager)
            print("y_train:")
            print(Counter(self.y_train.data))
            print("balanced y_train:")
            print(Counter(balanced_y_train.data))

            y_pred = pipeline.predict(self.X_test)
            score = accuracy_score(self.y_test.data, y_pred)
            end = time()
            print("score:", score)
            print("time:", end - start)
            self.assertGreater(score, 0.6)
            print('\n' * 2)

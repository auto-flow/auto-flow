#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
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
from autoflow.workflow.components.classification.random_forest import RandomForestClassifier
from autoflow.workflow.components.preprocessing.text.tokenize.simple import SimpleTokenlizer
from autoflow.workflow.components.preprocessing.text.topic.lda import LdaTransformer
from autoflow.workflow.components.preprocessing.text.topic.lsi import LsiTransformer
from autoflow.workflow.components.preprocessing.text.topic.nmf import NmfTransformer
from autoflow.workflow.components.preprocessing.text.topic.rp import RpTransformer
from autoflow.workflow.components.preprocessing.text.topic.tsvd import TsvdTransformer
from autoflow.workflow.ml_workflow import ML_Workflow


class TestIterAlforithm(LocalResourceTestCase):

    def test_classifier(self):
        train_df = datasets.load("titanic")[["Name", "Survived"]]
        y = np.array(train_df.pop("Survived"))

        X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.2,
                                                            random_state=0)
        X_train = DataFrameContainer("TrainSet", dataset_instance=X_train, resource_manager=self.mock_resource_manager)
        X_test = DataFrameContainer("TestSet", dataset_instance=X_test, resource_manager=self.mock_resource_manager)
        y_train = NdArrayContainer("TrainLabel", dataset_instance=y_train, resource_manager=self.mock_resource_manager)
        y_test = NdArrayContainer("TestLabel", dataset_instance=y_test, resource_manager=self.mock_resource_manager)
        X_train.set_feature_groups(["text"])
        X_test.set_feature_groups(["text"])
        est_cls_list = [
            TsvdTransformer,
            NmfTransformer,
            LsiTransformer,
            LdaTransformer,
            RpTransformer,
        ]
        for cls in est_cls_list:
            print("=========================")
            print(cls.__name__)
            print("=========================")
            tokenizer = SimpleTokenlizer(
                **get_default_hp_of_cls(SimpleTokenlizer)
            )
            tokenizer.in_feature_groups = "text"
            tokenizer.out_feature_groups = "token"
            transformer = cls(
                **get_default_hp_of_cls(cls)
            )
            transformer.in_feature_groups = "token"
            transformer.out_feature_groups = "num"
            classifier = RandomForestClassifier(
                **get_default_hp_of_cls(RandomForestClassifier)
            )
            pipeline = ML_Workflow(
                [
                    ("tokenizer", tokenizer),
                    ("transformer", transformer),
                    ("classifier", classifier),
                ],
                resource_manager=self.mock_resource_manager
            )
            start = time()
            pipeline.fit(X_train, y_train, X_test, y_test)
            y_pred = pipeline.predict(X_test)
            score = accuracy_score(y_test.data, y_pred)
            end = time()
            print("score:", score)
            print("time:", end - start)
            self.assertGreater(score, 0.6)
            print('\n' * 2)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from time import time

from autoflow.data_container import DataFrameContainer, NdArrayContainer
from autoflow.datasets import load
from autoflow.hdl.utils import get_default_hp_of_cls
from autoflow.tests.base import LocalResourceTestCase
from autoflow.workflow.components.preprocessing.reduce.fast_ica import FastICA
from autoflow.workflow.components.preprocessing.reduce.kernel_pca import KernelPCA
from autoflow.workflow.components.preprocessing.reduce.pca import PCA


def very_close(i, j, delta=2):
    if abs(i - j) <= delta:
        return True
    return False


class RunReduce(LocalResourceTestCase):
    def setUp(self) -> None:
        super(RunReduce, self).setUp()
        self.L = 1024
        df = load("qsar")
        y = df.pop("target")
        X = df
        X[X == 0] = -1
        X.index = reversed(X.index)
        self.index = deepcopy(X.index)
        X = DataFrameContainer("TrainSet", dataset_instance=X)
        X.set_feature_groups(["num"] * X.shape[1])
        X2 = deepcopy(X)
        y2 = deepcopy(y)
        N = 500
        X2.data = X2.data.iloc[:N, :]
        X2.set_feature_groups(["num"] * X2.shape[1])
        y2 = y2.iloc[:N]
        self.Xs = [
            X, X2
        ]
        self.ys = [
            NdArrayContainer("TrainLabel", dataset_instance=y),
            NdArrayContainer("TrainLabel", dataset_instance=y2)
        ]

    def calc_sp1(self, shape, ratio):
        return max(1, min(shape[0], round(shape[1] * ratio)))

    def test(self):
        for cls in [
            PCA,
            FastICA,
            KernelPCA
        ]:
            print("=========================")
            print(cls.__name__)
            print("=========================")
            for idx in [1]:
                for p in [0, 0.25, 0.5, 1]:
                    hp = get_default_hp_of_cls(cls)
                    hp.update({
                        "in_feature_groups": "num",
                        "out_feature_groups": "reduced",
                        "_n_components__sp1_ratio": p
                    })
                    start = time()
                    reducer = cls(**hp)
                    X = self.Xs[idx]
                    y = self.ys[idx]
                    reduced = reducer.fit_transform(X, y)["X_train"]
                    assert very_close(reduced.shape[1], self.calc_sp1(X.shape, p), delta=0)
                    print("consuming time :", time() - start)
                    print("assign ratio :", p)
                    print("actual ratio :", reduced.shape[1] / X.shape[1])
                    print("origin shape :", X.shape)
                    print("actual shape :", reduced.shape)
                    print("\n" * 2)


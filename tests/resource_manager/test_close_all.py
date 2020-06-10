#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, ShuffleSplit

from autoflow.constants import CONNECTION_POOL_CLOSE_MSG, START_SAFE_CLOSE_MSG, \
    END_SAFE_CLOSE_MSG, RESOURCE_MANAGER_CLOSE_ALL_LOGGER
from autoflow.core.classifier import AutoFlowClassifier
from autoflow.tests.base import LogTestCase


class TestCloseAll(LogTestCase):
    visible_levels = ("INFO", "WARNING")
    log_name = "test_close_all.log"
    def test_close_all(self):
        # todo : 增加预测与集成学习的案例
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        pipe = AutoFlowClassifier(
            DAG_workflow={
                "num->target": ["liblinear_svc", "libsvm_svc", "logistic_regression"]
            },
            initial_runs=2,
            run_limit=2,
            debug=True,
            log_file=self.log_file,
            resource_manager=self.mock_resource_manager
        )
        pipe.fit(
            X_train, y_train,
            splitter=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
            fit_ensemble_params=False
            # fixme: 目前不支持对hold out验证的集成学习
        )
        # score = accuracy_score(y_test, y_pred)
        score = pipe.score(X_test, y_test)
        print(score)
        # ----analyzing-----
        stack_cnt = 0
        self.update_log_path(pipe)
        for (level, logger, msg) in self.iter_log_items():
            if logger==RESOURCE_MANAGER_CLOSE_ALL_LOGGER:
                print("MESSAGE :",msg)
                if msg == START_SAFE_CLOSE_MSG.strip():
                    stack_cnt += 1
                elif msg == END_SAFE_CLOSE_MSG.strip():
                    stack_cnt -= 1
                elif msg == CONNECTION_POOL_CLOSE_MSG.strip():
                    if stack_cnt > 0:
                        pass
                    else:
                        raise Exception  # be completely wrapped

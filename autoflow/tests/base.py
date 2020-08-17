#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import os
import re
import shutil
import unittest
from pathlib import Path
from typing import Iterator, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder

from autoflow.datasets import load_task
from autoflow.tests.mock import get_mock_resource_manager


class LocalResourceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super(LocalResourceTestCase, self).setUp()
        self.mock_resource_manager = get_mock_resource_manager()

    def tearDown(self) -> None:
        shutil.rmtree(self.mock_resource_manager.store_path)


class LogTestCase(LocalResourceTestCase):
    visible_levels = None
    log_name = None

    def setUp(self) -> None:
        super(LogTestCase, self).setUp()
        self.log_file = os.getcwd() + "/" + self.log_name
        self.pattern = re.compile("\[(" + "|".join(self.visible_levels) + ")\]\s\[.*:(.*)\](.*)$", re.MULTILINE)
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def update_log_path(self, pipe):
        pipe.resource_manager.init_experiment_table()
        experiment = pipe.resource_manager.ExperimentModel
        log_path = experiment.select().where(experiment.experiment_id == pipe.experiment_id)[0].log_path
        self.log_file = log_path

    def iter_log_items(self) -> Iterator[Tuple[str, str, str]]:
        '''
        iterate log items
        Returns
        -------
        result:Iterator[Tuple[str,str,str]]
        (level, logger, msg)
        like: "INFO", "peewee", "SELECT * FROM table;"
        '''
        log_content = Path(self.log_file).read_text()

        for item in self.pattern.finditer(log_content):
            level = item.group(1)
            logger = item.group(2)
            msg = item.group(3)
            msg = msg.strip()
            yield (level, logger, msg)


class EstimatorTestCase(unittest.TestCase):
    current_file = None

    def setUp(self) -> None:
        cur_dir = Path(self.current_file).parent
        if (cur_dir / "126025.bz2").exists():
            X_train, y_train, X_test, y_test, cat = joblib.load(cur_dir / "126025.bz2")
        else:
            X_train, y_train, X_test, y_test, cat = load_task(126025)
            joblib.dump(
                [X_train, y_train, X_test, y_test, cat],
                cur_dir / "126025.bz2"
            )
        nan_cnt = np.count_nonzero(pd.isna(pd.concat([X_train, X_test])), axis=0)
        cat = np.array(cat)
        cat_na_mask = (nan_cnt > 0) & cat
        num_na_mask = (nan_cnt > 0) & (~cat)
        cat_imputer = SimpleImputer(strategy="constant", fill_value="NA").fit(X_train.loc[:, cat_na_mask])
        # num_imputer = BaseImputer(strategy="median").fit(X_train.loc[:, num_na_mask])
        X_train.loc[:, cat_na_mask] = cat_imputer.transform(X_train.loc[:, cat_na_mask])
        X_test.loc[:, cat_na_mask] = cat_imputer.transform(X_test.loc[:, cat_na_mask])
        # X_train.loc[:, num_na_mask] = num_imputer.transform(X_train.loc[:, num_na_mask])
        # X_test.loc[:, num_na_mask] = num_imputer.transform(X_test.loc[:, num_na_mask])
        ordinal_encoder = OrdinalEncoder(dtype="int").fit(X_train.loc[:, cat])
        transformer = StandardScaler().fit(X_train.loc[:, ~cat])
        X_train.loc[:, cat] = ordinal_encoder.transform(X_train.loc[:, cat])
        X_train.loc[:, ~cat] = transformer.transform(X_train.loc[:, ~cat])
        X_test.loc[:, cat] = ordinal_encoder.transform(X_test.loc[:, cat])
        X_test.loc[:, ~cat] = transformer.transform(X_test.loc[:, ~cat])
        self.cat_indexes = np.arange(len(cat))[cat]
        label_encoder = LabelEncoder().fit(y_train)
        self.y_train = label_encoder.transform(y_train)
        self.y_test = label_encoder.transform(y_test)
        self.X_train = X_train
        self.X_test = X_test

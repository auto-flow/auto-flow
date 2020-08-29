#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy

import numpy as np
import pandas as pd

from autoflow.feature_engineer.encode.entity import EntityEncoder
from autoflow.tests.base import EstimatorTestCase
from autoflow.utils.logging_ import setup_logger

setup_logger()


class TestEntityEncoder(EstimatorTestCase):
    current_file = __file__

    def test(self):
        X_train_copied = deepcopy(self.X_train)
        cols = self.X_train.columns[self.cat_indexes].tolist()
        entity_encoder = EntityEncoder(max_epoch=10, cols=cols).fit(self.X_train, self.y_train)
        transformed = entity_encoder.transform(self.X_test)
        print(transformed)
        assert np.all(self.X_train == X_train_copied)
        s = pd.Series(['age', 'workclass_0', 'workclass_1', 'workclass_2', 'fnlwgt',
       'education_0', 'education_1', 'education_2', 'education_3',
       'education-num', 'marital-status_0', 'marital-status_1', 'occupation_0',
       'occupation_1', 'occupation_2', 'occupation_3', 'relationship_0',
       'relationship_1', 'race_0', 'race_1', 'sex_0', 'capital-gain',
       'capital-loss', 'hours-per-week', 'native-country_0',
       'native-country_1', 'native-country_2', 'native-country_3',
       'native-country_4'])
        assert np.all(s == transformed.columns)
        assert np.all(transformed.index == self.X_test.index)
        # print(transformed.columns)

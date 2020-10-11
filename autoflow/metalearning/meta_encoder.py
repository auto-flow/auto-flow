#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.ambo.utils import ConfigurationTransformer
from autoflow.feature_engineer.encode import EntityEncoder
import pandas as pd
import numpy as np

def train_meta_encoder(encoder_params, config_space, X, y, results):
    entity_encoder = EntityEncoder(**encoder_params)
    config_transformer = ConfigurationTransformer(impute=None, encoder=entity_encoder)
    config_transformer.fit(config_space)
    config_transformer.fit_encoder(X, y)
    config_transformer.encoder.samples_db = [pd.DataFrame(), np.array([])]
    results.append(config_transformer)
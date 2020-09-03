#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
from joblib import load

from autoflow.opt.config_generators.density_estimator.tpe import TreeStructuredParzenEstimator
from autoflow.opt.utils import ConfigurationTransformer

config_space = load("config_space.bz2")
config_space_transformer = ConfigurationTransformer(impute=None, encoder=False)
config_space_transformer.fit(config_space)
tpe = TreeStructuredParzenEstimator()

tpe.set_config_transformer(config_space_transformer)
samples = config_space.sample_configuration(40)
vectors = config_space_transformer.transform(np.array([sample.get_array() for sample in samples]))
losses = np.random.rand(40)
tpe.fit(vectors, losses)
gen_samples = tpe.sample()
print(gen_samples)
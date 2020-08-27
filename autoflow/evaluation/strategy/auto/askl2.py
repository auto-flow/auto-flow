import json
import os
import pickle
from time import time

import numpy as np
import pandas as pd

from autoflow.utils.logging_ import get_logger
from .selector import OneVSOneSelector

logger = get_logger(__package__)
this_directory = os.path.abspath(os.path.dirname(__file__))
selector_file = os.path.join(this_directory, 'selector.pkl')
training_data_file = os.path.join(this_directory, 'askl2_training_data.json')
with open(training_data_file) as fh:
    training_data = json.load(fh)
metafeatures = pd.DataFrame(training_data['metafeatures'])
y_values = np.array(training_data['y_values'])
strategies = training_data['strategies']
minima_for_methods = training_data['minima_for_methods']
maxima_for_methods = training_data['maxima_for_methods']
if not os.path.exists(selector_file):
    logger.info("start training evaluation strategy selector ... ")
    start_time = time()
    selector = OneVSOneSelector(
        configuration=training_data['configuration'],
        default_strategy_idx=strategies.index('RF_SH-eta4-i_holdout_iterative_es_if'),
        rng=1,
    )
    selector.fit(
        X=metafeatures,
        y=y_values,
        methods=strategies,
        minima=minima_for_methods,
        maxima=maxima_for_methods,
    )
    with open(selector_file, 'wb') as fh:
        pickle.dump(selector, fh)
    cost_time = time() - start_time
    logger.info(f"finish training, cost {cost_time:.3f}s")
else:
    logger.info("start loading evaluation strategy selector ... ")
    start_time = time()
    with open(selector_file, "rb") as f:
        selector = pickle.load(f)
    cost_time = time() - start_time
    logger.info(f"finish loading, cost {cost_time:.3f}s")

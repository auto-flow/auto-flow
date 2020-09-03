#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from time import time, sleep

import numpy as np
import pandas as pd
import pynisher

from autoflow.metalearning.metafeatures.metafeatures import calculate_all_metafeatures_with_labels, \
    calculate_all_metafeatures_encoded_labels
from autoflow.utils.logging_ import get_logger
from autoflow.utils.ml_task import MLTask

logger = get_logger("Metafeaures")

EXCLUDE_META_FEATURES_CLASSIFICATION = {
    'Landmark1NN',
    'LandmarkDecisionNodeLearner',
    'LandmarkDecisionTree',
    'LandmarkLDA',
    'LandmarkNaiveBayes',
    'LandmarkRandomNodeLearner',
    'PCAFractionOfComponentsFor95PercentVariance',
    'PCAKurtosisFirstPC',
    'PCASkewnessFirstPC',
    'PCA'
}

EXCLUDE_META_FEATURES_REGRESSION = {
    'Landmark1NN',
    'LandmarkDecisionNodeLearner',
    'LandmarkDecisionTree',
    'LandmarkLDA',
    'LandmarkNaiveBayes',
    'LandmarkRandomNodeLearner',
    'PCAFractionOfComponentsFor95PercentVariance',
    'PCAKurtosisFirstPC',
    'PCASkewnessFirstPC',
    'NumberOfClasses',
    'ClassOccurences',
    'ClassProbabilityMin',
    'ClassProbabilityMax',
    'ClassProbabilityMean',
    'ClassProbabilitySTD',
    'ClassEntropy',
    'LandmarkRandomNodeLearner',
    'PCA',
}


def _calculate_metafeatures(feature_groups, x_train, y_train, ml_task: MLTask, dataset_name="default"):
    categorical = [feature_group in ("cat", "highC_cat") for feature_group in feature_groups]

    EXCLUDE_META_FEATURES = EXCLUDE_META_FEATURES_CLASSIFICATION \
        if ml_task.mainTask == "classification" else EXCLUDE_META_FEATURES_REGRESSION

    result = calculate_all_metafeatures_with_labels(
        x_train, y_train, categorical=categorical, dataset_name=dataset_name,
        dont_calculate=EXCLUDE_META_FEATURES)
    for key in list(result.metafeature_values.keys()):
        if result.metafeature_values[key].type_ != 'METAFEATURE':
            del result.metafeature_values[key]

    return result


def _calculate_metafeatures_encoded(feature_groups, x_train, y_train, ml_task: MLTask, dataset_name="default"):
    categorical = [feature_group in ("cat", "highC_cat") for feature_group in feature_groups]
    EXCLUDE_META_FEATURES = EXCLUDE_META_FEATURES_CLASSIFICATION \
        if ml_task.mainTask == "classification" else EXCLUDE_META_FEATURES_REGRESSION

    result = calculate_all_metafeatures_encoded_labels(
        x_train, y_train, categorical=categorical, dataset_name=dataset_name,
        dont_calculate=EXCLUDE_META_FEATURES)
    for key in list(result.metafeature_values.keys()):
        if result.metafeature_values[key].type_ != 'METAFEATURE':
            del result.metafeature_values[key]
    return result


def calculate_metafeatures(feature_groups: pd.Series, x_train: np.ndarray, y_train: np.ndarray, ml_task: MLTask,
                           memory_limit, dataset_name="default"):
    # basic metafeatures
    result1 = _calculate_metafeatures(feature_groups, x_train, y_train, ml_task, dataset_name)
    start_time = time()
    result2 = None
    time_limit = 5  # can not exceed 5s
    # advance metafeatures(skew ...)
    try:
        safe_mf = pynisher.enforce_limits(mem_in_mb=memory_limit,
                                          wall_time_in_s=time_limit,
                                          grace_period_in_s=time_limit,  # fixme : am I right?
                                          logger=logger)(
            _calculate_metafeatures_encoded)
        result2 = safe_mf(feature_groups, x_train, y_train, ml_task, dataset_name)
    except Exception as e:
        logger.error('Error getting metafeatures (encoded) : %s', str(e))
    cost_time = time() - start_time
    logger.info(f"_calculate_metafeatures_encoded cost time: {cost_time:.3f}s")
    if result2 is not None:
        result1.metafeature_values.update(result2.metafeature_values)
    metafeatures = result1
    data_ = {mf.name: mf.value for mf in metafeatures.metafeature_values.values()}
    return data_


if __name__ == '__main__':
    from autoflow.datasets import load
    from autoflow import AutoFlowClassifier

    df = load("titanic")
    autoflow = AutoFlowClassifier(evaluation_strategy="simple")
    autoflow.fit(X_train=df, column_descriptions={"target": "Survived"}, is_not_realy_run=True)
    result = calculate_metafeatures(
        autoflow.data_manager.feature_groups,
        autoflow.data_manager.X_train.data.values,
        autoflow.data_manager.y_train.data,
        autoflow.ml_task,
        1024*15,
        "A"
    )
    print(autoflow)

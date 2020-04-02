import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from hyperflow.estimator.base import AutoPipelineEstimator
from hyperflow.hdl.hdl_constructor import HDL_Constructor
from hyperflow.manager.resource_manager import ResourceManager
from hyperflow.tuner.tuner import Tuner

path = ("../data/train_0312.csv")


# data preprocessing
def data_preprocessing():
    train_data_path = path
    df = pd.read_csv(train_data_path)

    # 填充空值
    df['RO5_violations'].fillna(value=0, inplace=True)
    df['AlogP'].fillna(value=df['AlogP'].mean(), inplace=True)

    # 处理列表特征
    dffeature = df.Features.values
    feat_list = list(dffeature)
    newfeature = []
    for line in dffeature:
        line = line[1:-1]
        arra = line.split(',')
        arra = np.array(arra).astype(np.float64)
        newfeature.append(arra)

    # 特征合并
    newfeatureall = np.concatenate(
        (np.array(df.Molecule_max_phase).reshape(6924, 1), np.array(df.AlogP).reshape(6924, 1)), axis=1)
    newfeature = np.concatenate((newfeature, newfeatureall), axis=1)

    # 均值填充NaN
    for i in range(newfeature.shape[1]):
        ind = np.isnan(newfeature[:, i])
        if ind.any():
            temp = newfeature[:, i]
            a = temp[~np.isnan(temp)].mean()
            newfeature[:, i][np.isnan(temp)] = a

    # 标准化
    stdScale = StandardScaler().fit(newfeature)
    newfeaturenorm = stdScale.transform(newfeature)

    # 区间化
    bins = [-9, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 24]
    new_range = pd.cut(df.Label, bins)
    newlabel = np.array(df.Label)
    return newfeaturenorm, newlabel, new_range


x_train, y_train, y_range = data_preprocessing()

tuner = Tuner(
    initial_runs=12,
    run_limit=120,
)
hdl_constructor = HDL_Constructor(
    DAG_descriptions={
        "num->target": "lightgbm"
    }
)
resource_manager = ResourceManager(os.getcwd() + "/for_hxw_result")
hyperflow_pipeline = AutoPipelineEstimator(tuner, hdl_constructor, ensemble_builder=False)

hyperflow_pipeline.fit(
    X=x_train, y=y_train, n_jobs=3
)
joblib.dump(hyperflow_pipeline, "hyperflow_pipeline_for_hxw.bz")

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import ShuffleSplit

from autoflow.estimator.base import AutoFlowEstimator
from autoflow.hdl.hdl_constructor import HDL_Constructor
from autoflow.tuner import Tuner

boston = load_boston()
data = boston.data
target = boston.target
columns = list(boston.feature_names)+ ["target"]
df = pd.DataFrame(np.hstack([data,target[:,None]]), columns=columns)
ss = ShuffleSplit(n_splits=1, random_state=0, test_size=0.25)
train_ix, test_ix = next(ss.split(df))
df_train = df.iloc[train_ix, :]
df_test = df.iloc[test_ix, :]

tuner = Tuner(
    initial_runs=5,
    run_limit=12,
)
autoflow_pipeline = AutoFlowEstimator(tuner, HDL_Constructor(
    DAG_workflow={
        "num->num": [
            "select.from_model_reg","select.univar_reg","select.rfe_reg"#,None
        ],
        "num->target": [
            "lightgbm"
        ]
    }
))
column_descriptions = {
    "target": "target"
}
autoflow_pipeline.fit(
    X_train=df_train, X_test=df_test, column_descriptions=column_descriptions, n_jobs=1
)

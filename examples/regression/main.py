import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from hyperflow import HyperFlowRegressor

train_df = pd.read_csv("../data/train_regression.csv")
train_df.replace("NA", np.nan, inplace=True)
test_df = pd.read_csv("../data/test_regression.csv")
test_df.replace("NA", np.nan, inplace=True)
trained_pipeline = HyperFlowRegressor(initial_runs=5, run_limit=10, n_jobs=1, included_regressors=["lightgbm"],
                                      per_run_time_limit=60)
column_descriptions = {
    "id": "Id",
    "target": "SalePrice",
}
if not os.path.exists("hyperflow_regression.bz2"):
    trained_pipeline.fit(
        X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
        should_store_intermediate_result=True,
        splitter=KFold(n_splits=3, shuffle=True, random_state=42), fit_ensemble_params=False
    )
    joblib.dump(trained_pipeline, "hyperflow_regression.bz2")
predict_pipeline = joblib.load("hyperflow_regression.bz2")
result = predict_pipeline.predict(test_df)
print(result)

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from autoflow import AutoFlowRegressor

train_df = pd.read_csv("../data/train_regression.csv")
train_df.replace("NA", np.nan, inplace=True)
test_df = pd.read_csv("../data/test_regression.csv")
test_df.replace("NA", np.nan, inplace=True)
trained_pipeline = AutoFlowRegressor(initial_runs=5, run_limit=10, n_jobs=1, included_regressors=["lightgbm"],
                                      per_run_time_limit=60)
column_descriptions = {
    "id": "Id",
    "target": "SalePrice",
}
if not os.path.exists("autoflow_regression.bz2"):
    trained_pipeline.fit(
        X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
        splitter=KFold(n_splits=3, shuffle=True, random_state=42), fit_ensemble_params=False
    )
    # if you want to see the workflow AutoFlow is searching, you can use `draw_workflow_space` to visualize
    hdl_constructor = trained_pipeline.hdl_constructors[0]
    hdl_constructor.draw_workflow_space()
    joblib.dump(trained_pipeline, "autoflow_regression.bz2")
predict_pipeline = joblib.load("autoflow_regression.bz2")
result = predict_pipeline.predict(test_df)
print(result)

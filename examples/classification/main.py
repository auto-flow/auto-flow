import joblib
import pandas as pd
import os
from hyperflow import HyperFlowClassifier

train_df = pd.read_csv("../data/train_classification.csv")
test_df = pd.read_csv("../data/test_classification.csv")
trained_pipeline = HyperFlowClassifier(initial_runs=5, run_limit=10, n_jobs=1, included_classifiers=["lightgbm"])
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}
if not os.path.exists("hyperflow_classification.bz2"):
    trained_pipeline.fit(
        X=train_df, X_test=test_df, column_descriptions=column_descriptions, should_store_intermediate_result=True
    )
    joblib.dump(trained_pipeline, "hyperflow_classification.bz2")
predict_pipeline = joblib.load("hyperflow_classification.bz2")
result = predict_pipeline.predict(test_df)

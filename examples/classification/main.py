import os

import joblib
import pandas as pd
from sklearn.model_selection import KFold

from hyperflow import HyperFlowClassifier

# load data from csv file
train_df = pd.read_csv("../data/train_classification.csv")
test_df = pd.read_csv("../data/test_classification.csv")
# initial_runs  -- is totally random search, to provide experience for SMAC algorithm
# run_limit     -- is the maximum number of runs
# n_jobs        -- start how many search process
# included_classifiers -- restrict the search space . lightgbm is the only classifier that needs to be selected
# per_run_time_limit -- restrict the run time. if a trial during 60 seconds, it is expired, should be shut off
trained_pipeline = HyperFlowClassifier(initial_runs=5, run_limit=10, n_jobs=1, included_classifiers=["lightgbm"],
                                       per_run_time_limit=60)
# describing meaning of columns. `id`, `target` and `ignore` all has specific meaning
# `id` is a column name means unique descriptor of each rows,
# `target` is column in the dataset is what your model will learn to predict
# `ignore` is some columns which contains irrelevant information
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}
if not os.path.exists("hyperflow_classification.bz2"):
    # seed `train_df`, `test_df` and `column_descriptions` to classifier,
    # if param `fit_ensemble_params` set as "auto", Stack Ensemble will be used
    # `splitter` is the "train dataset" and "valid dataset" split method, in here it is set as 3-Fold Cross Validation
    trained_pipeline.fit(
        X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
        fit_ensemble_params=False,
        splitter=KFold(n_splits=3, shuffle=True, random_state=42),
    )
    # finally , the best model will be serialize and store in local file system for subsequent use
    joblib.dump(trained_pipeline, "hyperflow_classification.bz2")
    # if you want to see the workflow HyperFlow is searching, you can use `draw_workflow_space` to visualize
    hdl_constructor = trained_pipeline.hdl_constructors[0]
    hdl_constructor.draw_workflow_space()
# suppose you are processing predict procedure, firstly, you should load serialized model from file system
predict_pipeline = joblib.load("hyperflow_classification.bz2")
# secondly, use loaded model to do predicting
result = predict_pipeline.predict(test_df)
print(result)

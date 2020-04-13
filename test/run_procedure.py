from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import KFold

import hyperflow
from hyperflow import HyperFlowClassifier, DataManager

examples_path = Path(hyperflow.__file__).parent.parent / "examples"
train_df = pd.read_csv(examples_path / "data/train_classification.csv")
test_df = pd.read_csv(examples_path / "data/test_classification.csv")
pipe = HyperFlowClassifier(initial_runs=5, run_limit=10, n_jobs=1, included_classifiers=["lightgbm"])
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}
pipe.data_manager = DataManager(X_train=train_df, X_test=test_df, column_descriptions=column_descriptions)
pipe.hdl_constructors[0].run(pipe.data_manager, pipe.random_state, pipe.highR_cat_threshold)
pipe.hdl_constructors[0].draw_workflow_space()

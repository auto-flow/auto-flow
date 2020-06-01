import os
from pathlib import Path

import pandas as pd

import autoflow
from autoflow import AutoFlowClassifier, DataManager

examples_path = Path(autoflow.__file__).parent.parent / "examples"
train_df = pd.read_csv(examples_path / "data/train_classification.csv")
test_df = pd.read_csv(examples_path / "data/test_classification.csv")
pipe = AutoFlowClassifier(initial_runs=5, run_limit=10, n_jobs=1, included_classifiers=["lightgbm"])
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}
pipe.data_manager = DataManager(X_train=train_df, X_test=test_df, column_descriptions=column_descriptions)
pipe.hdl_constructors[0].run(pipe.data_manager, pipe.random_state, pipe.highR_cat_threshold)
graph = pipe.hdl_constructors[0].draw_workflow_space()
open("workflow_space.gv").write(graph.source)
cmd = f'''dot -Tpng -Gsize=9,15\! -Gdpi=300 -oworkflow_space.png workflow_space.gv'''
os.system(cmd)

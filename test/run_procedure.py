import pandas as pd
from sklearn.model_selection import ShuffleSplit

from hyperflow.estimator.base import AutoPipelineEstimator
from hyperflow.tuner.tuner import Tuner

df = pd.read_csv("../examples/classification/train_classification.csv")
ss = ShuffleSplit(n_splits=1, random_state=0, test_size=0.25)
train_ix, test_ix = next(ss.split(df))
df_train = df.iloc[train_ix, :]
df_test = df.iloc[test_ix, :]

tuner = Tuner(
    initial_runs=5,
    run_limit=12,
)
hyperflow_pipeline = AutoPipelineEstimator(tuner)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}
hyperflow_pipeline.fit(
    X=df_train, X_test=df_test, column_descriptions=column_descriptions, n_jobs=1
)

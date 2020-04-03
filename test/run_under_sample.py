import pandas as pd
from sklearn.model_selection import ShuffleSplit

from hyperflow.estimator.base import HyperFlowEstimator
from hyperflow.hdl.hdl_constructor import HDL_Constructor
from hyperflow.tuner.tuner import Tuner

df = pd.read_csv("../examples/classification/train_classification.csv")
ss = ShuffleSplit(n_splits=1, random_state=0, test_size=0.25)
train_ix, test_ix = next(ss.split(df))
df_train = df.iloc[train_ix, :]
df_test = df.iloc[test_ix, :]

tuner = Tuner(
    initial_runs=30,
    run_limit=0,
)
hdl_constructor = HDL_Constructor(
    DAG_descriptions={
        "nan->imp": "impute.fill_abnormal",
        "imp->{cat_name=cat,num_name=num}": "operate.split.cat_num",
        "cat->num": "encode.cat_boost",
        "over_sample": [
            "balance.under_sample.all_knn",
            "balance.under_sample.cluster_centroids",
            "balance.under_sample.condensed_nearest_neighbour",
            "balance.under_sample.edited_nearest_neighbours",
            "balance.under_sample.instance_hardness_threshold",
            "balance.under_sample.near_miss",
            "balance.under_sample.neighbourhood_cleaning_rule",
            "balance.under_sample.one_sided_selection",
            "balance.under_sample.random",
            "balance.under_sample.repeated_edited_nearest_neighbours",
            "balance.under_sample.tomek_links",
        ],
        "num->target": {"_name": "lightgbm", "_vanilla": True}
    }
)
hyperflow_pipeline = HyperFlowEstimator(tuner, hdl_constructor)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}

hyperflow_pipeline.fit(
    X=df_train, X_test=df_test, column_descriptions=column_descriptions, n_jobs=1
)

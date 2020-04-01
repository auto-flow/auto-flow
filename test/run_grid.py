import pandas as pd
from sklearn.model_selection import ShuffleSplit

from hyperflow.estimator.base import AutoPipelineEstimator
from hyperflow.hdl.hdl_constructor import HDL_Constructor
from hyperflow.tuner.smac_tuner import Tuner

df = pd.read_csv("../examples/classification/train_classification.csv")
ss = ShuffleSplit(n_splits=1, random_state=0, test_size=0.25)
train_ix, test_ix = next(ss.split(df))
df_train = df.iloc[train_ix, :]
df_test = df.iloc[test_ix, :]

hdl_constructor = HDL_Constructor(
    DAG_descriptions={
        "nan->{highR=highR_nan,lowR=lowR_nan}": "operate.split.nan",
        "lowR_nan->nan": "impute.fill_abnormal",
        "highR_nan->nan": "operate.drop",
        "all->{cat_name=cat,num_name=num}": "operate.split.cat_num",
        "cat->num": ["encode.cat_boost", "encode.target", "encode.label"],
        "num->target": {"_name": "lightgbm", "_vanilla": True}
    }
)
tuner = Tuner(
    run_limit=-1,
    search_method="grid"
)
hyperflow_pipeline = AutoPipelineEstimator(tuner, hdl_constructor)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}

hyperflow_pipeline.fit(
    X=df_train, X_test=df_test, column_descriptions=column_descriptions, n_jobs=1
)

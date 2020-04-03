import pandas as pd

from hyperflow.estimator.base import HyperFlowEstimator
from hyperflow.hdl.hdl_constructor import HDL_Constructor
from hyperflow.tuner.tuner import Tuner

df = pd.read_csv("../data/QSAR.csv")

hdl_constructor = HDL_Constructor(
    DAG_descriptions={
        "num->var": "compress.variance",
        "var->pea": {"_name": "compress.pearson", "n_jobs": 6},
        "pea->target": "logistic_regression"
    }
)
tuner = Tuner(
    run_limit=12,
    initial_runs=12,
    search_method="smac"
)
hyperflow_pipeline = HyperFlowEstimator(tuner, hdl_constructor)
column_descriptions = {
    "id": "Name",
    "target": "labels"
}

hyperflow_pipeline.fit(
    X=df, column_descriptions=column_descriptions, n_jobs=3
)

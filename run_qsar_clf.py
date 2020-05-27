#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import os
from pathlib import Path

# from autoflow.datasets import load
import joblib
from sklearn.model_selection import KFold

import autoflow
from autoflow import HDL_Constructor, AutoFlowClassifier
from autoflow.tuner import Tuner

try:
    import click
except Exception:
    os.system("pip install click")
    import click


@click.command()
@click.option("--file", "-f", help="input file name", type=click.Path())
@click.option("--inst", "-i", help="instance id", default=0, type=int)
@click.option("--target", "-t", help="target column name", type=str)
@click.option("--ignore", "-g", help="ignore column name", type=str, multiple=True)
@click.option("--id", help="id column name", type=str)
@click.option("--n_jobs", "-n", default=5, help="n_jobs_in_algorithm", type=str)
def main(file, inst, target, ignore, id, n_jobs):
    if inst is None:
        inst = 0
    examples_path = Path(autoflow.__file__).parent.parent / "data"
    # 加载pickle格式的数据
    train_df = joblib.load(examples_path / file)
    # train_df = load("qsar")
    estimators = [
        "adaboost",
        "decision_tree",
        "extra_trees",
        "k_nearest_neighbors",
        "liblinear_svc",
        "libsvm_svc",
        "lightgbm"
        "logistic_regression"
        "random_forest"
        "sgd"
    ]

    hdl_constructor = HDL_Constructor(
        DAG_workflow={
            "num->selected": {
                "_name": "select.from_model_clf",
                "_vanilla": True,
                "estimator": "sklearn.ensemble.ExtraTreesClassifier",
                "n_estimators": 10,
                "max_depth": 7,
                "min_samples_split": 10,
                "min_samples_leaf": 10,
                "random_state": inst,
                "n_jobs": 1,
                "_select_percent": {"_type": "quniform", "_value": [10, 60, 1], "_default": 40}
            },
            "selected->target": estimators
        }
    )

    hdl_constructors = [hdl_constructor] * 2

    tuners = [
        Tuner(
            search_method="random",
            search_method_params={
                "specific_allocate": {
                    ("estimating", "adaboost"): 1,
                    ("estimating", "decision_tree"): 2,
                    ("estimating", "extra_trees"): 5,
                    ("estimating", "k_nearest_neighbors"): 1,
                    ("estimating", "liblinear_svc"): 1,
                    ("estimating", "libsvm_svc"): 1,
                    ("estimating", "lightgbm"): 10,
                    ("estimating", "logistic_regression"): 5,
                    ("estimating", "random_forest"): 5,
                    ("estimating", "sgd"): 2,
                }
            },
            # run_limit=50,
            per_run_time_limit=1200,
            per_run_memory_limit=30 * 1024,
            n_jobs_in_algorithm=n_jobs
        ),
        Tuner(
            search_method="smac",
            run_limit=100,
            initial_runs=1,
            debug=False,
            per_run_time_limit=1200,
            per_run_memory_limit=30 * 1024,
            n_jobs_in_algorithm=n_jobs
        )
    ]

    trained_pipeline = AutoFlowClassifier(
        consider_ordinal_as_cat=False,
        tuner=tuners,
        hdl_constructor=hdl_constructors,
        random_state=inst,
        db_type="postgresql", db_params={
            "user": "postgres",
            "host": "123.56.90.56",
            "port": 5432,
            "password": "xenon",
        }
    )
    column_descriptions = {
        # "id": "Name",
        # "ignore": ["Smiles", "labels"],
        # "target": "pIC50"
    }
    if id:
        column_descriptions.update({"id": id})
    if ignore:
        column_descriptions.update({"ignore": list(ignore)})
    column_descriptions.update({"target": target})
    # if not os.path.exists("autoflow_classification.bz2"):
    trained_pipeline.fit(
        X_train=train_df, column_descriptions=column_descriptions,
        splitter=KFold(n_splits=5, shuffle=True, random_state=0),
        fit_ensemble_params=False,
        dataset_metadata={
            "name": file
        },
        task_metadata={
            "instance_id": inst
        },
        specific_task_token=str(inst)
        # is_not_realy_run=True
    )
    print(trained_pipeline)


if __name__ == '__main__':
    main()
Examples
===========
You can find every data files used in examples in ``examples/data``.

Automatic Classification
---------------------------
.. code-block:: console

    $ cd examples/classification
    $ python main.py

**First step**, import several needed packages.

.. code-block:: python

    import os

    import joblib
    import pandas as pd
    from sklearn.model_selection import KFold

    from autoflow import AutoFlowClassifier

**Second step**, load data from CSV.

.. code-block:: python

    train_df = pd.read_csv("../data/train_classification.csv")
    test_df = pd.read_csv("../data/test_classification.csv")

**Third step**, define a ``AutoFlowClassifier``.

Here are some key parameters:
    * ``initial_runs``  are totally random search, to provide experience for SMAC algorithm.
    * ``run_limit`` is the maximum number of runs.
    * ``n_jobs`` defines how many search processes are started.
    * ``included_classifiers`` restrict the search space . In here ``lightgbm`` is the only classifier that needs to be selected. You can use ``included_classifiers=["lightgbm", "random_forest"]`` to define other selected classifiers. You can find all classifiers AutoFlow supported in :class:`autoflow.hdl.hdl_constructor.HDL_Constructor`
    * ``per_run_time_limit`` restrict the run time. if a trial during 60 seconds, it is expired, should be killed.

.. code-block:: python

    trained_pipeline = AutoFlowClassifier(initial_runs=5, run_limit=10, n_jobs=1, included_classifiers=["lightgbm"],
                                           per_run_time_limit=60)

**Fifth step**, define columns descriptions, you can find .You can find the full definition in :class:`autoflow.manager.data_manager.DataManager` .

Here are some columns descriptions:
    * ``id`` is a column name means unique descriptor of each rows.
    * ``target`` column in the dataset is what your model will learn to predict.
    * ``ignore`` is some columns which contains irrelevant information.

.. code-block:: python

    column_descriptions = {
        "id": "PassengerId",
        "target": "Survived",
        "ignore": "Name"
    }

**Sixth step**, auto do fitting. you can find full document in :meth:`autoflow.estimator.base.AutoFlowEstimator.fit` .

Passing data params ``train_df``, ``test_df`` and ``column_descriptions`` to classifier.

If ``fit_ensemble_params`` is "auto" or True, the top 10 models will be integrated by stacking ensemble.

``splitter`` is train-valid-dataset splitter,now is set to ``KFold(3, True, 42)`` to do 3-Fold Cross-Validation.

You can pass this param defined by yourself or other package, like :class:`sklearn.model_selection.StratifiedKFold`.

.. code-block:: python

    trained_pipeline.fit(
        X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
        fit_ensemble_params=False,
        splitter=KFold(n_splits=3, shuffle=True, random_state=42),
    )

**Finally**, the best model will be serialize and store in local file system for subsequent use.

.. code-block:: python

    joblib.dump(trained_pipeline, "autoflow_classification.bz2")

**Additionally**, if you want to see what the workflow AutoFlow is searching,
you can use :meth:`autoflow.hdl.hdl_constructor.HDL_Constructor#draw_workflow_space` to visualize.

>>> hdl_constructor = trained_pipeline.hdl_constructors[0]
>>> hdl_constructor.draw_workflow_space()

.. image:: images/workflow_space.png

**For Reproducibility purpose**, you can load serialized model from file system.

.. code-block:: python

    predict_pipeline = joblib.load("autoflow_classification.bz2")
    result = predict_pipeline.predict(test_df)

OK, you can do automatically classify now.


Automatic Regression
---------------------------

.. code-block:: console

    $ cd examples/regression
    $ python main.py


.. code-block:: python

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

Phased Search
---------------------------

``phased search`` allowed you to search the best pipeline(workflow) step by step.

For example, if you want do ``feature selection``, there are several variable you should take into consideration.
If you search these hyper-parameters in one search task, the search-space can be very huge.

Without loss of generality, we hypothesis the search-space contains two hyper-parameters :
:math:`A` and :math:`B` , if :math:`A` and :math:`B` are both categorical-type hyper-param,
:math:`A` have :math:`M` choices, :math:`B` have :math:`N` choices, the total hyper-parameters space have :math:`{N}\times{M}` choices,
time complexity is :math:`{O}({A}\times{B})`.

But if we greedily fine tune :math:`A` in the first, and then fine tune :math:`B` on the optimal :math:`A`, we can
we can reduce time complexity to :math:`O({A} + {B})` .

This is a very simple greedy thought, but it is very powerful and make sense if there is little correlation between each variable.

In AutoFlow, we support such search method by ``phased search``, you should define a list of :class:`autoflow.hdl.hdl_constructor.HDL_Constructor` and a list of
:class:`autoflow.tuner.tuner.Tuner`, and AutoFlow will run each ``hdl_constructor`` and ``tuner`` .
After finish one of phases, the best configuration will be extracted, and passing the intersted super parameter into the next ``HDL`` to cover ``placeholder``.

Back to ``feature selection`` example, there are almost two kinds of hyper-parameters: ``feature selection algorithm`` and
``feature selection ratio``.

In the first step, we should keep ``feature selection ratio`` is same and change ``feature selection algorithm``.

.. code-block:: python

    DAG_workflow={  # There are 4 algorithms to choice
        "num->selected": [
            {"_name": "select.from_model_clf", "_select_percent": 80},
            {"_name": "select.rfe_clf", "_select_percent": 80},
        ],
        "selected->target": {"_name": "logistic_regression", "_vanilla": True}
        # _vanilla means don't update hyper-params from hdl_bank to keep
        # there are no hyper-params in classifier.
    }

In second step, we should choose the best ``feature selection algorithm`` searched from first step and to find optimal
``feature selection ratio``.

.. code-block:: python

    DAG_workflow={
    # selector algorithm is set to "<placeholder>",
    # means this variable will use the optimal value on the same position in the previous
    # search will be selected
        "num->selected": {"_name": "<placeholder>",
                     "_select_percent": {"_type": "quniform", "_value": [1, 100, 0.5],
                                         "_default": 80}},
        # instead of fixed "_select_percent" hyper-param, in this phase it will be fine tuned.
        "selected->target": {"_name": "logistic_regression", "_vanilla": True}
    }


**Hear is the complete code**

.. code-block:: console

    $ cd examples
    $ python phased_search.py


.. code-block:: python

    import pandas as pd

    from autoflow.estimator.base import AutoFlowEstimator
    from autoflow.hdl.hdl_constructor import HDL_Constructor
    from autoflow.tuner.tuner import Tuner

    df_train = pd.read_csv("./data/train_classification.csv")

    hdl_constructors = [
        HDL_Constructor(
            DAG_workflow={
                "nan->{highR=highR_nan,lowR=lowR_nan}": "operate.split.nan",
                "lowR_nan->nan": "impute.fill_abnormal",
                "highR_nan->nan": "operate.drop",
                "all->{cat_name=cat,num_name=num}": "operate.split.cat_num",
                "cat->num": "encode.label",
                "num->selected": [
                    {"_name": "select.from_model_clf", "_select_percent": 80},
                    {"_name": "select.rfe_clf", "_select_percent": 80},
                ],
                "selected->target": {"_name": "logistic_regression", "_vanilla": True}
            }
        ),
        HDL_Constructor(
            DAG_workflow={
                "nan->{highR=highR_nan,lowR=lowR_nan}": "operate.split.nan",
                "lowR_nan->nan": "impute.fill_abnormal",
                "highR_nan->nan": "operate.drop",
                "all->{cat_name=cat,num_name=num}": "operate.split.cat_num",
                "cat->num": "encode.label",
                "num->selected": {"_name": "<placeholder>",
                             "_select_percent": {"_type": "quniform", "_value": [1, 100, 0.5],
                                                 "_default": 80}},
                "selected->target": {"_name": "logistic_regression", "_vanilla": True}
            }
        ),
    ]

    tuners = [
        Tuner(
            run_limit=-1,
            search_method="grid",
            n_jobs=3,
            debug=True
        ),
        Tuner(
            run_limit=50,
            initial_runs=10,
            search_method="smac",
            n_jobs=3,
            debug=True
        ),
    ]
    autoflow_pipeline = AutoFlowEstimator(tuners, hdl_constructors)
    column_descriptions = {
        "id": "PassengerId",
        "target": "Survived",
        "ignore": "Name"
    }

    autoflow_pipeline.fit(
        X_train=df_train, column_descriptions=column_descriptions
    )

Intruduction
============



Tabular Data-Processing Workflow
----------------------------------

`Titanic <https://www.kaggle.com/c/titanic>`_ is perhaps the most familiar machine learning task for data scientists.
The sub table after sampling is shown below:

.. csv-table:: Titanic Origin Data
   :file: csv/origin.csv

You must notice that such raw table cannot be used in data-mining or machine-learning.
We should fill the missing value, encoding the category value, and so on.

In order to introduce the practical problems this project wants to solve,
I want to introduce a concept: ``feature group``.

Feature Group
~~~~~~~~~~~~~~~~~

Except the columns that cannot provide entity specific properties, like ``id``,
the remaining columns are called ``features``.

.. note::
    You can find ``column_descriptions``'s definition in :class:`autoflow.manager.data_manager.DataManager`

If some ``features`` have similar properties, they are containing in a same ``feature group``.

.. note::
    You can find some ``feature group``'s examples and practices in :class:`autoflow.hdl.hdl_constructor.HDL_Constructor`


nan
:::::::::
``nan`` is abbreviation of ``Not a Number``, indicating that this column has missing value, like this:

>>> from numpy import NaN
>>> import pandas as pd
>>> import numpy as np
>>> column = [1, 2, 3, 4, NaN]

num
:::::::::
``num`` is abbreviation of ``numerical``, indicating that this column are all numerical value.

.. note::
    Only ``num`` `feature group` can used in estimating phase

For example:

>>> column = [1, 2, 3, 4, 5]

cat
::::::::::
``cat`` is abbreviation of ``categorical``, indicating  this column has `any` string-type value.

For example:

>>> column = [1, 2, 3, 4, "a"]

num_nan
::::::::::
``num_nan`` is abbreviation of ``numerical NaN``, indicating this column is full of numbers except for missing values.

For example:

>>> column = [1, 2, 3, 4, NaN]

cat_nan
::::::::::
``cat_nan`` is abbreviation of ``categorical NaN``, indicating this  column has at least one string other than the missing value.

For example:

>>> column = [1, 2, 3, "a", NaN]

highR_nan
::::::::::
``highR_nan`` is abbreviation of ``high ratio NaN``, indicating this  column has most of this column is missing.

For example:

>>> column = [1, 2, NaN, NaN, NaN]
>>> np.count_nonzero(pd.isna(column)) / column.size
0.6

NaN ratio is 0.6, more than 0.5 (default highR_nan_threshold)

lowR_nan
::::::::::
``highR_nan`` is abbreviation of ``high ratio NaN``, indicating this this column has most of this column is missing.

For example:

>>> column = [1, 2, 3, NaN, NaN]
>>> np.count_nonzero(pd.isna(column)) / column.size
0.4

NaN ratio is 0.4, less than 0.5 (default ``highR_nan_threshold``)

highR_cat
::::::::::
``highR_cat`` is abbreviation of ``high cardinality ratio categorical``, indicating this this column is a categorical column (see in :ref:`cat`),
and the unique value of this column divided by the total number of this column is more than ``highR_cat_threshold`` .

For example:

>>> column = ["a", "b", "c", "d", "d"]
>>> rows = len(column)
>>> np.unique(column).size / rows
0.8

cardinality ratio is 0.8, more than 0.5 (default ``highR_cat_threshold``)

lowR_cat
::::::::::
``lowR_cat`` is abbreviation of ``low cardinality ratio categorical``, indicating this this column is a categorical column (see in :ref:`cat`),
and the unique value of this column divided by the total number of this column is less than ``lowR_cat_threshold`` .

For example:

>>> column = ["a", "b", "d", "d", "d"]
>>> rows = len(column)
>>> np.unique(column).size / rows
0.4

cardinality ratio is 0.8, less than 0.5 (default ``lowR_cat_threshold``)




Work Flow
~~~~~~~~~~

After defining a concept: ``feature group``, ``Workflow`` is the next important concept.

You can consider the whole machine-learning training and testing procedure as a directed acyclic graph(DAG),
except ETL or other data prepare and feature extract procedure.

In this graph , you can consider nodes as ``feature group``,
edges as `data-processing or estimating algorithms`.
Each edges' tail node is a ``feature group`` **before processing**,
each edges' head node is a other ``feature group`` **after processing**.

You should keep in mind that, each edge represents **one** algorithm or **a list of**
algorithms. For example, after a series of data-processing, single :ref:`num` (numerical)
`feature group` is reserved, we should do estimating(`fit features to target column`):

.. graphviz::

   digraph estimating {
      "num" -> "target" [ label="{lightgbm, random_forest}" ];
   }

In this figure we can see: ``lightgbm`` and ``random_forest`` are candidate algorithms.
Some computer scientists said, ``AutoML`` is a ``CASH`` problem (Combined Algorithm Selection and Hyper-parameter optimization problem).

In fact, the algorithm selection on the edge allows this ``workflow`` to be called a ``workflow space``.

Hear is the ``workflow space`` figure for `Titanic <https://www.kaggle.com/c/titanic>`_  task.

.. image:: images/workflow_space.png

Instance In Titanic
~~~~~~~~~~~~~~~~~~~~
You may be curious about the ``workflow space`` picture above, want to know how it work.
Let me introduce the processing details step by step.

**First step**, data manager(:class:`autoflow.manager.data_manager.DataManager`) split raw data into three
``feature group``: :ref:`nan`, :ref:`highR_nan`, :ref:`cat` and :ref:`num`. like this:

.. csv-table:: First Step : Split By Data Manager
   :file: csv/origin_split.csv

This corresponds to this figure:

.. graphviz::

   digraph estimating {
      "data" -> "cat" [ label="data_manager: cat" ];
      "data" -> "num" [ label="data_manager: num" ];
      "data" -> "nan" [ label="data_manager: nan" ];
      "data" -> "highR_nan" [ label="data_manager: highR_nan" ];
   }

-----------------------------------------------------------------------------------

**Second step**, ``highR_nan_imputer`` should process :ref:`highR_nan` `feature group` to
:ref:`nan`, ``merge`` (means don't do any thing, just rename the `feature group`) and
``drop`` are candidate algorithms. in this case, we choose ``drop`` option.

.. csv-table:: Second Step : process highR nan
   :file: csv/process_highR_nan.csv

This corresponds to this figure:

.. graphviz::

   digraph estimating {
      "data" -> "cat" [ label="data_manager: cat" ];
      "data" -> "num" [ label="data_manager: num" ];
      "data" -> "nan" [ label="data_manager: nan" ];
      "data" -> "highR_nan" [ label="data_manager: highR_nan" ];
      "highR_nan" -> "nan" [ label="{operate.drop, operate.merge}" ];
   }

-----------------------------------------------------------------------------------

**Third step**, using ``operate.split.cat_num`` algorithm to split :ref:`nan` to two
feature groups: :ref:`cat_nan` and :ref:`num_nan`.

.. csv-table:: Third Step : Split Categorical and Numerical
   :file: csv/split_cat_num.csv

This corresponds to this figure:

.. graphviz::

   digraph estimating {
      "data" -> "cat" [ label="data_manager: cat" ];
      "data" -> "num" [ label="data_manager: num" ];
      "data" -> "nan" [ label="data_manager: nan" ];
      "data" -> "highR_nan" [ label="data_manager: highR_nan" ];
      "highR_nan" -> "nan" [ label="{operate.drop, operate.merge}" ];
      "nan" -> "cat_nan" [ label="operate.split.cat_num: cat_nan" ];
      "nan" -> "num_nan" [ label="operate.split.cat_num: num_nan" ];
   }

-----------------------------------------------------------------------------------

**Fourth step**, fill :ref:`cat_nan` to :ref:`cat`, fill :ref:`num_nan` to :ref:`num`.

.. csv-table:: Fourth Step : Fill NaN
   :file: csv/fill.csv

This corresponds to this figure:

.. graphviz::

   digraph estimating {
      "data" -> "cat" [ label="data_manager: cat" ];
      "data" -> "num" [ label="data_manager: num" ];
      "data" -> "nan" [ label="data_manager: nan" ];
      "data" -> "highR_nan" [ label="data_manager: highR_nan" ];
      "highR_nan" -> "nan" [ label="{operate.drop, operate.merge}" ];
      "nan" -> "cat_nan" [ label="operate.split.cat_num: cat_nan" ];
      "nan" -> "num_nan" [ label="operate.split.cat_num: num_nan" ];
      "cat_nan" -> "cat" [ label="impute.fill_cat" ];
      "num_nan" -> "num" [ label="impute.fill_num" ];
   }

-----------------------------------------------------------------------------------

**Fifth step**, using ``operate.split.cat`` algorithm to split :ref:`cat` to two
feature groups: :ref:`highR_cat` and :ref:`lowR_cat`.

.. csv-table:: Fifth Step : Split Categorical to :ref:`highR_cat` and :ref:`lowR_cat`
   :file: csv/split_cat.csv

This corresponds to this figure:

.. graphviz::

   digraph estimating {
      "data" -> "cat" [ label="data_manager: cat" ];
      "data" -> "num" [ label="data_manager: num" ];
      "data" -> "nan" [ label="data_manager: nan" ];
      "data" -> "highR_nan" [ label="data_manager: highR_nan" ];
      "highR_nan" -> "nan" [ label="{operate.drop, operate.merge}" ];
      "nan" -> "cat_nan" [ label="operate.split.cat_num: cat_nan" ];
      "nan" -> "num_nan" [ label="operate.split.cat_num: num_nan" ];
      "cat_nan" -> "cat" [ label="impute.fill_cat" ];
      "num_nan" -> "num" [ label="impute.fill_num" ];
      "cat" -> "highR_cat" [ label="operate.split.cat: highR_cat" ];
      "cat" -> "lowR_cat" [ label="operate.split.cat: lowR_cat" ];
   }


-----------------------------------------------------------------------------------


**Sixth step**, we encode :ref:`highR_cat` to :ref:`num` by ``label_encoder``,
we encode :ref:`lowR_cat` to :ref:`num` by ``one_hot_encoder``,

.. csv-table:: Sixth Step : Encoding Categorical to Numerical
   :file: csv/encode.csv

This corresponds to this figure:

.. graphviz::

   digraph estimating {
      "data" -> "cat" [ label="data_manager: cat" ];
      "data" -> "num" [ label="data_manager: num" ];
      "data" -> "nan" [ label="data_manager: nan" ];
      "data" -> "highR_nan" [ label="data_manager: highR_nan" ];
      "highR_nan" -> "nan" [ label="{operate.drop, operate.merge}" ];
      "nan" -> "cat_nan" [ label="operate.split.cat_num: cat_nan" ];
      "nan" -> "num_nan" [ label="operate.split.cat_num: num_nan" ];
      "cat_nan" -> "cat" [ label="impute.fill_cat" ];
      "num_nan" -> "num" [ label="impute.fill_num" ];
      "cat" -> "highR_cat" [ label="operate.split.cat: highR_cat" ];
      "highR_cat" -> "num" [ label="encode.label" ];
      "cat" -> "lowR_cat" [ label="operate.split.cat: lowR_cat" ];
      "lowR_cat" -> "num" [ label="encode.one_hot" ];
   }

-----------------------------------------------------------------------------------

**Seventh step**, finally, we finish all the data preprocessing phase,
now we should do estimating. ``lightgbm`` and ``random_forest`` are candidate algorithms.

This corresponds to this figure:

.. graphviz::

   digraph estimating {
      "data" -> "cat" [ label="data_manager: cat" ];
      "data" -> "num" [ label="data_manager: num" ];
      "data" -> "nan" [ label="data_manager: nan" ];
      "data" -> "highR_nan" [ label="data_manager: highR_nan" ];
      "highR_nan" -> "nan" [ label="{operate.drop, operate.merge}" ];
      "nan" -> "cat_nan" [ label="operate.split.cat_num: cat_nan" ];
      "nan" -> "num_nan" [ label="operate.split.cat_num: num_nan" ];
      "cat_nan" -> "cat" [ label="impute.fill_cat" ];
      "num_nan" -> "num" [ label="impute.fill_num" ];
      "cat" -> "highR_cat" [ label="operate.split.cat: highR_cat" ];
      "highR_cat" -> "num" [ label="encode.label" ];
      "cat" -> "lowR_cat" [ label="operate.split.cat: lowR_cat" ];
      "lowR_cat" -> "num" [ label="encode.one_hot" ];
      "num" -> "target" [ label="{lightgbm, random_forest}" ];
   }




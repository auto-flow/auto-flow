==========
HyperFlow
==========


In the problem of data mining and machine learning of tabular data,
data scientists usually group the features, construct a directed acyclic graph (DAG),
and form a machine learning workflow.

In each directed edge of this directed acyclic graph, 
the tail node represents the feature group before preprocessing, 
and the head node represents the feature group after preprocessing. 
Edge representation data processing or feature engineering , in each edge algorithm selection and hyper-parameter optimization are doing.

Unfortunately, if data scientists want to manually select algorithms and hyper-parameters for such a workflow, 
it will be a very tedious task. In order to solve this problem, 
we developed the ``Hyperflow``, 
which can automatically select algorithm and optimize the parameters of machine learning workflow. 
In other words, it can implement AutoML of table data.

.. image:: docs/_images/workflow_space.png







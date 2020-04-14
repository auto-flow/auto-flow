==========
HyperFlow
==========

Introduction
--------------

In the problem of data mining and machine learning of tabular data,
data scientists usually group the features, construct a directed acyclic graph (DAG),
and form a machine learning workflow.

In each directed edge of this directed acyclic graph, 
the tail node represents the feature group before preprocessing, 
and the head node represents the feature group after preprocessing. 
Edge representation data processing or feature engineering algorithms, 
in each edge algorithm selection and hyper-parameter optimization are doing.

Unfortunately, if data scientists want to manually select algorithms and 
hyper-parameters for such a workflow, 
it will be a very tedious task. In order to solve this problem, 
we developed the ``Hyperflow``, 
which can automatically select algorithm and optimize the parameters of 
machine learning workflow. 
In other words, it can implement AutoML of table data.

.. image:: docs/_images/workflow_space.png


Installation
--------------

Requirements
~~~~~~~~~~~~~~

This project is built and test on Linux system, so Linux platform is required. 
If you are using Windows system, `WSL <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ is worthy of considerarion.

Besides the listed requirements (see requirements.txt), the `random forest <https://github.com/automl/random_forest_run>`_ 
used in `SMAC3 <https://github.com/automl/SMAC3>`_ requires 
`SWIG <http://www.swig.org/>`_ (>= 3.0, <4.0) as a build dependency. 
If you are using Ubuntu or another Debain Linux, you can enter following command :

::

    apt-get install swig

On Arch Linux (or any distribution with swig4 as default implementation):

::

    pacman -Syu swig3
    ln -s /usr/bin/swig-3 /usr/bin/swig

HyperFlow re







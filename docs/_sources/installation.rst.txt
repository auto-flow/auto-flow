Installation
============

.. _requirements:

System Requirements
------------

HyperFlow has the following system requirements:

  * Linux operating system (for example Ubuntu),
  * Python (>=3.6).
  * C++ compiler (with C++11 supports) and SWIG (version 3.0 or later)

Besides the listed requirements (see requirements.txt), the `random forest <https://github.com/automl/random_forest_run>`_
used in `SMAC3 <https://github.com/automl/SMAC3>`_ requires
`SWIG <http://www.swig.org/>`_ (>= 3.0, <4.0) as a build dependency.

To install the C++ compiler and SWIG system-wide on a linux system with apt,
please call:

.. code-block:: bash

    sudo apt-get install build-essential swig

If you use Anaconda, you have to install both, gcc and SWIG, from Anaconda to
prevent compilation errors:

.. code-block:: bash

    conda install gxx_linux-64 gcc_linux-64 swig

.. _installation_pypi:

Installation from pypi
----------------------
To install HyperFlow from pypi, please use the following command on the command
line:

.. code-block:: bash

    pip install HyperFlow
    
If you want to install it in the user space (e.g., because of missing
permissions), please add the option :code:`--user` or create a virtualenv.

.. _manual_installation:

Manual Installation
-------------------
To install HyperFlow from command line, please type the following commands on the
command line:

.. code-block:: bash

    git clone https://github.com/Hyper-Flow/HyperFlow.git
    cd HyperFlow
    python setup.py install

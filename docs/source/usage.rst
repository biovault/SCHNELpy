Usage
====================

In this section, the basic usage of the schnelpy package is provided.

Importing the package
---------------------

In order to use the package in the python script, at the top of .py file write

.. code-block:: python

   import schnel

Loading sample data
--------------------

To load sample test data, the user of the package can write the following line of code:

.. code-block:: python

    x, y = schnel.DataPrep.loaddata.loadmnist()

..

where **x** returns images of MNIST test set (10 thousand records) and **y** returns labels of
MNIST test set.


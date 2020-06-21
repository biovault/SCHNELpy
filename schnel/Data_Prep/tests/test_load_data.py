from nose.tools import assert_list_equal
import numpy as np
from schnel.Data_Prep.load_data import load_mnist
import os.path

class test_dataprep(object):

    def test_load_data(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        X_load, Y_load = load_mnist()
        pathX = os.path.join(my_path, "../../data/test_load_dataX.npy")
        pathY = os.path.join(my_path, "../../data/test_load_dataY.npy")
        assertX = np.load(pathX)
        assertY = np.load(pathY)
        assert_list_equal(list(X_load.flatten()), list(assertX.flatten()))
        assert_list_equal(list(Y_load.flatten()), list(assertY.flatten()))

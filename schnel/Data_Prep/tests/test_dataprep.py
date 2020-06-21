from nose.tools import assert_list_equal
import numpy as np
from schnel.Data_Prep.dataprep import parse_to_numpy
import os.path


class test_dataprep(object):
    def test_parse_to_numpy_default(self):
        np_arr = np.arange(2)
        transformed_arr = np_arr
        ret_arr = parse_to_numpy(np_arr)
        assert_list_equal(list(transformed_arr.flatten()),list(ret_arr.flatten()))

    def test_parse_to_numpy_log(self):
        np_arr = np.arange(2)
        transformed_arr = np.log(np_arr+1)
        ret_arr = parse_to_numpy(np_arr, transformation='log')
        assert_list_equal(list(transformed_arr.flatten()), list(ret_arr.flatten()))

    def test_parse_to_numpy_arcsinh(self):
        np_arr = np.arange(2)
        transformed_arr = np.true_divide(np_arr, 5)
        transformed_arr = np.arcsinh(transformed_arr)
        ret_arr = parse_to_numpy(np_arr, transformation='arcsinh')
        assert_list_equal(list(transformed_arr.flatten()), list(ret_arr.flatten()))

    def test_parse_to_numpy_csv(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        np_arr = np.array([0,1,3,0,1,2,0,2,3,0,1,2])
        source = os.path.join(my_path, "../../data/test_csv.csv")
        ret_arr = parse_to_numpy(source)
        assert_list_equal(list(np_arr.flatten()), list(ret_arr.flatten()))

    def test_parse_to_numpy_fcs(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        source = os.path.join(my_path, "../../data/test_fcs_to_csv_data.fcs")
        ret_arr = parse_to_numpy(source)
        np_source = os.path.join(my_path, "../../data/test_fcs.npy")
        np_arr = np.load(np_source)
        assert_list_equal(list(np_arr.flatten()), list(ret_arr.flatten()))

    def test_parse_to_numpy_h5ad(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        source = os.path.join(my_path, "../../data/pbmc3k.h5ad")
        ret_arr = parse_to_numpy(source)
        np_source = os.path.join(my_path, "../../data/test_h5ad.npy")
        np_arr = np.load(np_source)
        assert_list_equal(list(np_arr.flatten()), list(ret_arr.flatten()))

    








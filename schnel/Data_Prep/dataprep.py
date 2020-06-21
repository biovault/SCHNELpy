import os
from schnel.Data_Prep.csv_to_numpy import csv_to_numpy as ctn
from schnel.Data_Prep.fcs_to_numpy import fcs_to_numpy as ftn
from schnel.Data_Prep.h5ad_to_numpy import h5ad_to_numpy as htn
from schnel.Data_Prep.pca import pca
import scanpy as sc
import numpy as np
import anndata as ad



def parse_to_numpy(source, transformation=None, cofactor=5, features_after_pca=50):
    """
    Main method that accepts files of type: .csv, .fcs, .h5ad as wel as objects of type: np.ndarray and h5ad object. It
    also transforms(log, arcsinh) data or performs pca analysis on it.

    :param source: file/object to become an ndarray
    :param transformation: one of two can be applied 'log' - logistic, 'arcsinh' - arcsine, Default is set to None
    :param cofactor: only use when applying arcsinh transformation, Default is set to 5
    :param features_after_pca: amount of features to stay after performing pca on given data, default is 50
    :return: an np.ndarray ready to be clustered
    """
    #pylint: disable=unused-variable

    if cofactor == 0:
        print("scale cannot be 0")
    elif isinstance(source, sc.AnnData):
        np_arr = htn(source, features_after_pca)
    elif isinstance(source, np.ndarray):
        np_arr = source
    else:
        file_name, file_extension = os.path.splitext(source)
        np_arr = None
        if file_extension == ".csv":
            np_arr = ctn(source)
        elif file_extension == ".fcs":
            np_arr = ftn(source)
        elif file_extension == ".h5ad":
            np_arr = htn(source, features_after_pca)
        else:
            print("file type: " + file_extension + " not recognized by parser.\n "
                                                   "Acceptable types are: .csv, .fcs, .h5ad")

    if np.size(np_arr) > features_after_pca:
        np_arr = pca(np_arr, features_after_pca)

    transformed = np_arr
    if transformation == "arcsinh":
        divided = np.true_divide(np_arr, cofactor)
        transformed = np.arcsinh(divided)
    elif transformation == "log":
        transformed = np.log(np_arr+1)

    return transformed


if __name__ == "__main__":
    data = ad.read_h5ad('../data/pbmc3k.h5ad')
    parse_to_numpy(data)
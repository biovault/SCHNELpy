#pylint: disable=import-error

from schnel.clustering.HSNE_parser import read_HSNE_binary
import math
import schnel.Data_Prep.dataprep as dp
import numpy_to_hsne
import numpy as np
import os


def cluster(source, feature_ids=None, num_of_scales=0, num_of_neighbours=30, transformation_method=None, cofactor=5,
            p_comps=None, cell_by_feature=True, csv_header=False):
    """

    This method is responsible for running the whole SCHNEL algorithm pipeline.
    Input data is parsed and transferred into a numpy array. Afterwards, the numpy array is processed with the help of C++ code
    wrapped with pybind11. The output is a hsne binary file containing the generated HSNE hierarchy.
    The HSNE hierarchy file is read by the HSNE parser, after which it is clustered with the Leiden algorithm.
    The output of this function is a list of matrices with columns as scales and rows as cluster classifications.

    :param source: file path, ndarray, h5ad object or list of file paths
    :param feature_ids: array of components on which the data should be clustered
    :param num_of_scales: number of scales used for creating an HSNE hierarchy structure
    :param num_of_neighbours: number of neighbours used in clustering
    :param transformation_method: type of transformation used (log, arch or None)
    :param cofactor: if arcsinh was specified you can pass the cofactor for thistransformation
    :param p_comps: Number of principal componenets after PCA
    :param cell_by_feature: true if input data is cell by feature, false if feature by cell
    :return: list of matrices equal to the size of the points/cells (rows)by the number of hierarchy scales (columns)
    """
    #pylint: disable=too-many-arguments

    #private parameters
    seeds = -1
    landmark_treshold = 1.5
    num_trees = 6
    num_checks = 1024
    trans_matrix_prune_treshold = 1.5
    num_walks = 200
    num_walks_per_landmark = 200
    monte_carlo_sampling = True
    out_of_core_computation = True


    #todo:
    #multiple files
    src_list = []
    ret_lens = []
    curr_len = 0
    if p_comps is None:
        p_comps = 50
    if isinstance(source, np.ndarray):
        np_arr = source
        ret_lens.append(len(np_arr))
    else:
        if isinstance(source, str):
            source = [source]
        for elem in source:
            np_elem = dp.parse_to_numpy(elem, transformation=transformation_method, cofactor=cofactor,
                                        features_after_pca=p_comps, csv_header=csv_header)
            ret_lens.append(len(np_elem) + curr_len)
            curr_len = curr_len + len(np_elem)
            src_list.append(np_elem)

        np_arr = np.vstack(src_list)
    if feature_ids is not None:
        np_arr = np_arr[:, feature_ids]
    check_num = int(math.log10(len(np_arr)/100))
    if num_of_scales < 2 and check_num > 1:
        num_of_scales = check_num
    elif num_of_scales < 2 and check_num < 2:
        print("The specified number of scales is below 2. The default value (2) will be used")
        num_of_scales = 2
    if cell_by_feature is False:
        np_arr = np_arr.transpose()
    if np.isnan(np_arr).any() or np.isinf(np_arr).any():
        print("Some of the fields in the data set are NaN or Inf")
        return
    numpy_to_hsne.run(np_arr, 'values.hsne', num_of_scales, seeds, landmark_treshold, num_of_neighbours,
                      num_trees, num_checks, trans_matrix_prune_treshold, num_walks, num_walks_per_landmark,
                      monte_carlo_sampling, out_of_core_computation)
    hsne = read_HSNE_binary('values.hsne')
    scaled_clusters = []
    for i in range(hsne.num_scales - 1):
        elem_clusters = hsne.cluster_scale(i + 1)
        elem_clusters = np.asarray(elem_clusters)
        scaled_clusters.append(elem_clusters.tolist())

    scaled_clusters = np.transpose(np.array(scaled_clusters))
    clusters_by_file = []
    prev = 0
    for length in ret_lens:
        clusters_by_file.append(scaled_clusters[prev:length][:])
        prev = length

    print("Clustering done")
    os.remove('values.hsne')
    print("Created clusters on ", hsne.num_scales, " scales..")
    return clusters_by_file
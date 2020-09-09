import anndata as ad
import scanpy as sc


def h5ad_to_numpy(source, pca=50, previous_pca=False):
    """
    Takes in a h5ad source, performs pca if specified and returns a numpy array of results.

    :param source: file path the h5ad file
    :param pca: desired PC's. Default 50.
    :param previous_pca: True to use objects previously computed PCA. Default false.
    :return: numpy.ndarray
    """
    if isinstance(source, str):
        data = ad.read_h5ad(source)
    else:
        data = source
    if previous_pca:
        try:
            return data.obsm['X_pca']
        except KeyError:
            print("Previous PCA not computed, set previous_pca to false")
    sc.tl.pca(data, n_comps=pca)
    return data.obsm['X_pca']

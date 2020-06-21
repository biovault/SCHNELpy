import sklearn.decomposition as sk


def pca(data, comps):
    """
    Returns pca of data with desired components.

    :param data: numpy array to perform PCA on
    :param comps: number of PC's
    :return: array with PCA transformation
    """
    pca_trans = sk.PCA(n_components=min(data.shape[0], min(data.shape[1], comps)))
    return pca_trans.fit_transform(data)

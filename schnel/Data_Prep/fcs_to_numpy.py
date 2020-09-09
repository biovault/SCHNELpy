import fcsparser as fcs

def fcs_to_numpy(file_path):
    """
    Translate an .fcs extension file data into a numpy array

    :param file_path: path to the file
    :return: numpy.ndarray
    """
    meta, data = fcs.parse(file_path, reformat_meta=True, meta_data_only=False)
    return data

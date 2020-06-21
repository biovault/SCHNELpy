import FlowCal


def fcs_to_numpy(file_path):
    """
    translate fcs extension file data into a numpy array

    :param file_path: path to the file
    :return: numpy.ndarray
    """
    s = FlowCal.io.FCSFile(file_path)
    return s.data

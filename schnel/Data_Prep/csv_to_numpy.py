from numpy import genfromtxt


def csv_to_numpy(file_path):
    """
    Translates csv files to numpy array.

    :param file_path: path to file
    :return: numpy array
    """
    return genfromtxt(file_path, delimiter=',')

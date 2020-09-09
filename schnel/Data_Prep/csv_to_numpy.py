from numpy import genfromtxt


def csv_to_numpy(file_path, csv_header=False):
    """
    Translates a csv file into a numpy array.

    :param csv_header: make true if there are column names in the data
    :param file_path: path to file
    :return: numpy array
    """
    if csv_header:
        return genfromtxt(file_path, delimiter=',', dtype=float, skip_header=True)
    return genfromtxt(file_path, delimiter=',', names=None, dtype=float)

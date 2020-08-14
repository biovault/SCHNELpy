import glob


def read_directory(directory_path, ext=""):
    """
    Reads multiple files from a directory.

    :param directory_path: directory from which files will be read.
    :param ext: extension of files that should be read from this directory. (simple regex is also recognized).
    :return: List with files ready to pass to main cluster function.
    """
    if directory_path[-1] != '/':
        directory_path += '/'
    directory_path += "*"
    file_path = directory_path + ext
    return glob.glob(file_path)

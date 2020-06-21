from mlxtend.data import loadlocal_mnist
import os.path

def load_mnist():
    """
    Sample MNIST data used to show basic functionality of SCHNEL package.

    :return: images of MNIST data set (10k), labels of MNIST data set (10k)
    """
    my_path = os.path.abspath(os.path.dirname(__file__))
    image_file = os.path.join(my_path, "../data/t10k-images.idx3-ubyte")
    labels_file = os.path.join(my_path, "../data/t10k-labels.idx1-ubyte")
    data, labels = loadlocal_mnist(
        images_path=image_file,
        labels_path=labels_file
    )
    return data, labels

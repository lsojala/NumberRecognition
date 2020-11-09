import mnist
import numpy


def load_mnist(dataset:str,path:str):
    """
    Retrives data from mnist data-files
    Args:
        dataset (str): dataset type, options are "train images", "train labels", "test imges", "test labels"
        path (str): path string to the folder of MNIST files.
    Returns numpy aray of crresponding data.
    """

    filename_pointer = {
        "train images": "train-images.idx3-ubyte",
        "train labels": "train-labels.idx1-ubyte",
        "test images": "t10k-images.idx3-ubyte",
        "test labels": "t10k-labels.idx1-ubyte"
        }

    filename = path + filename_pointer[dataset]

    with open(filename, 'rb') as file:
        data = mnist.parse_idx(file)

    return data
    
def encode_data(labeldata,hot_vector = 1):
        """
        Takes array of label data, and transforms it into array of one-hot encoded vectors.
        Args:
            labeldata (iterable): Array or list of data labels
            hot_vector: value assingned for hot, default 1
        Returns:
            numpy array of vectored labels
        """
        vec_data = []
        for label in labeldata:
            vector = [0] * 10
            for i in range(10):
                if label == i:
                    vector[i] = hot_vector                  
            vec_data.append(vector)

        data = numpy.array(vec_data)
        
        return data
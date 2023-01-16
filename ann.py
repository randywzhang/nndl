import numpy as np

"""
Basic ANN built by following Michael Nielsen's book Neural Networks and Deep Learning
"""
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        # initializes biases for all layers except the input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # randn creates the weight matrix with shape (y, x)
        # W*x -> (y, x) * (x, 1) -> (y, 1) shape
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    """
    Forward propagation of the network
    @param x - input to the network, in our case MNIST pixels
    
    assumes valid input
    """
    def feed_forward(self, x):
        for weight_matrix in self.weights:
            x = np.matmul(weight_matrix, x)

        return x


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


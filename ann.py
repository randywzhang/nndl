import numpy as np

# TODO: stochastic implementation

"""
Basic ANN built by following Michael Nielsen's book Neural Networks and Deep Learning

Network architecture : MLP, supervised learning
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
    @return - a tuple of the list of activations and list of z_values
              the activations list contains the output of each neuron in
              each layer and the z_values contain the inputs to each layer
              excluding the input layer
    
              activations[-1] contains the output vector of the network
    
    assumes valid input
    """
    def feed_forward(self, x):
        # The activation of each layer is f(z) where f is the sigmoid
        # function and z is the vector obtained by multiplying the
        # matrix of weights by the activation in the previous layer
        activations = [x]
        z_values = []
        for bias_vector, weight_matrix in zip(self.biases, self.weights):
            z = np.matmul(weight_matrix, x) + bias_vector
            x = sigmoid(z)
            z_values.append(z)
            activations.append(x)

        return activations, z_values

    """
    Back propagation algorithm
    Determines the gradient of the loss function with respect to each 
    parameter in the network and updates the weights by subtracting
    the gradient scaled down by a learning rate η (eta).
    
    @param x - input image
    @param y - expected output
    """
    def back_propagate(self, x, y):
        # learning rate η (eta)
        eta = 1e-5

        # derivatives of the loss function with respect to weights/biases
        dLdW = [np.zeros(w.shape) for w in self.weights]
        dLdb = [np.zeros(b.shape) for b in self.biases]

        # feed forward and record activations and z values
        activations, z_values = self.feed_forward(x)

        # TODO: calculate gradient

        # update network variables based on gradient
        self.weights -= dLdW * eta
        self.biases -= dLdb * eta

    """
    This network uses a quadratic loss function
    @param actual - network output for a given input x
    @param expected - labeled value for the given input x
    
    1/2 makes the derivatives nicer for backprop
    """
    def loss(self, actual, expected):
        return np.square(np.linalg.norm(actual - expected))/2

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


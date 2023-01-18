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
        # GRADIENT CALCULATION

        # begin with the derivative of the loss function with respect to the input
        # to the final layer
        dLdz = self.loss_derivative(activations[-1], y) * sigmoid_prime(z_values[-1])

        # dL/db = dL/dz * dz/db, dz/db = 1 => dL/db = dL/dz
        dLdb[-1] = dLdz
        # dL/dW = dL/dz * dz/dW = dL/dz * activation of previous layer
        # for each weight, change in z with respect to that weight is the activation
        # of the neuron in the previous layer.
        # dLdz_last has shape (y, 1) and activations[-2] has shape (x, 1) so to get the
        # matrix of dL/dW with shape (y, x) we need to transpose the activations and
        # multiply
        dLdW[-1] = np.matmul(dLdz, np.transpose(activations[-2]))

        # continue chaining derivatives until each variable has been updated
        for layer in range(2, self.num_layers - 1):
            # let z, denote the input to the layer before z
            # a denotes the activation of the layer before z
            # dL/dz, = dL/dz * dz/da * da/dz,
            # dz/da = W the weight matrix
            # da/dz, = sigmoid_prime(z,)
            # da/dz, has shape (x, 1), dL/dz has shape (y, 1) and W has shape (y, x)
            # we need dL/dz * dz/da to be the same shape as da/dz, so we transpose W
            # and matrix multiply with dL/dz
            dLdz = np.matmul(np.transpose(self.weights[-layer + 1]), dLdz) * sigmoid_prime(activations[-layer])
            dLdb[-layer] = dLdz  # as before
            dLdW[-layer] = np.matmul(dLdz, np.transpose(activations[-layer - 1]))  # as before

        # return gradients of network variables
        return dLdb, dLdW


def loss(actual, expected):
    """
    This network uses a quadratic loss function
    This function is unused in network training but is a good reference
    for learning purposes

    @param actual - network output for a given input x
    @param expected - labeled value for the given input x

    1/2 makes the derivatives nicer for backprop
    """
    return np.square(np.linalg.norm(actual - expected)) / 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# The following functions are derivatives used for backpropagation
def loss_derivative(actual, expected):
    # TODO: loss(actual, expected) == loss(expected, actual)
    # what happens when we return expected - actual?
    # expected - actual = -1 (actual - expected)
    # the partial derivatives change sign so the gradient also changes sign
    #
    # shouldn't it be np.linalg.norm(actual - expected) ??
    # no, because the norm is a scalar and we want each weight to change based
    # on how far from the expected value it is. We don't want each weight to
    # change the same amount.
    #
    # So why is it actual - expected and not expected - actual?
    # (x - x0)^2 == (x0 - x)^2
    # d/dx (x0 - x)^2 = 2 (x0 - x) * -1 <-- forgot this -1
    return actual - expected


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

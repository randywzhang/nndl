import numpy as np
from keras.datasets import mnist
import pickle

"""
Basic ANN built by following Michael Nielsen's book Neural Networks and Deep Learning
to classify handwritten digits from the MNIST dataset

Network architecture : MLP, supervised learning
"""
class Network(object):

    """
    Initializes network with random weights and biases taken from a normal distribution
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        # initializes biases for all layers except the input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # randn creates the weight matrix with shape (y, x) with values taken from
        # the gaussian distribution
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
        activation = x
        activations = [x]
        z_values = []
        for bias_vector, weight_matrix in zip(self.biases, self.weights):
            z = np.matmul(weight_matrix, activation) + bias_vector
            z_values.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        return activations, z_values

    """
    Back propagation algorithm
    Determines the gradient of the loss function with respect to each 
    parameter in the network
    
    @param x - input image
    @param y - expected output
    """
    def back_propagate(self, x, y):
        # derivatives of the loss function with respect to weights/biases
        dLdW = [np.zeros(w.shape) for w in self.weights]
        dLdb = [np.zeros(b.shape) for b in self.biases]

        # feed forward and record activations and z values
        activations, z_values = self.feed_forward(x)

        # GRADIENT CALCULATION

        # begin with the derivative of the loss function with respect to the input
        # to the final layer
        dLdz = loss_derivative(activations[-1], y) * sigmoid_prime(z_values[-1])

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
        for layer in range(2, self.num_layers):
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

    """
    Stochastic gradient descent implementation 
    Applies the backpropagation algorithm to the training data and updates 
    the network variables by subtracting the gradient scaled down by a 
    learning rate η (eta).
    @param training_data - labeled mnist images used to train the network
    @param epochs - number of times to repeatedly train on the training data
    @param batch_size - the size of each batch of training data
    @param learning_rate - the factor by which we scale the gradient down in 
        order to prevent divergence from overshooting the minima
    """
    def SGD(self, training_data, epochs, batch_size, learning_rate, test_data=None):
        for epoch in range(epochs):
            # shuffle the data, prevent overfitting
            np.random.shuffle(training_data)

            # split training data into batches
            batches = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]

            for batch in batches:
                stochastic_b = [np.zeros(b.shape) for b in self.biases]
                stochastic_W = [np.zeros(W.shape) for W in self.weights]

                # aggregate the gradients from each data point
                for training_datum in batch:
                    dLdb, dLdW = self.back_propagate(training_datum[0], training_datum[1])

                    # add the gradients together
                    stochastic_b = [b + db for b, db in zip(stochastic_b, dLdb)]
                    stochastic_W = [W + dW for W, dW in zip(stochastic_W, dLdW)]

                # update the network variables
                # by taking the average of all of the gradients by dividing by batch size
                # and scaling down by the learning rate
                # len(batch) is guaranteed to be batch_size for all batches except
                # batches[-1] where it could be less than the batch_size
                self.biases = [b - db * learning_rate / len(batch) for b, db in zip(self.biases, stochastic_b)]
                self.weights = [W - dW * learning_rate / len(batch) for W, dW in zip(self.weights, stochastic_W)]

            if test_data:
                # TODO: visualize network variables here
                # evaluate the network after every epoch
                accuracy = self.evaluate(test_data)
                print("Epoch {0}: {1}".format(epoch, accuracy))

    """
    Network evaluation function
    Takes test data as input and evaluates the network based on percentage
    of correct classifications
    @param test_data - test dataset to feed forward through the network
    @return - percentage of correctly classified images
    """
    def evaluate(self, test_data):
        # keep track of correct classifications
        num_correct = 0

        for datum in test_data:
            # send each input in a forward pass through the network
            activations, z_values = self.feed_forward(datum[0])

            # check that the activation of the final layer has its max value
            # at the labeled digit
            if np.argmax(activations[-1]) == datum[1]:
                num_correct += 1

        return num_correct / len(test_data)


"""
This network uses a quadratic loss function
This function is unused in network training but is a good reference
for learning purposes

@param actual - network output for a given input x
@param expected - labeled value for the given input x

1/2 makes the derivatives nicer for backprop
"""
def loss(actual, expected):
    return np.square(np.linalg.norm(actual - expected)) / 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# The following functions are derivatives used for backpropagation
"""
@param actual - n-d array denoting the activations of the final layer
@param expected - scalar denoting the expected digit
@return - n-d array denoting the difference between the actual result
    and basis vector ei with i = expected
"""
def loss_derivative(actual, expected):
    # (x - x0)^2 == (x0 - x)^2
    # d/dx (x0 - x)^2 = 2 (x0 - x) * -1 == 2 (actual - expected)

    # create basis vector
    expected_basis = np.zeros(actual.shape)
    expected_basis[expected] = 1.0

    return actual - expected_basis


def sigmoid_prime(x):
    # d/dx (1 / (1 + e^-x)) = e^-x / (1 + e^-x)^2
    #                       = (e^-x + 1 - 1) / (1 + e^-x)^2
    #                         factor out 1 / (1 + e^-x)
    #                       = [1 / (1 + e^-x)] * [(1 + e^-x - 1) / (1 + e^-x)]
    #                       = sigmoid(x) * [(1 + e^-x) / (1 + e^-x) - 1 / (1 + e^-x)]
    #                       = sigmoid(x) * (1 - sigmoid(x))
    return sigmoid(x) * (1 - sigmoid(x))


# TODO: implement saving and loading
"""
Utility functions for saving and loading a network
"""
def save_network(ann, filename):
    with open(filename, 'wb') as out_file:
        pickle.dump(ann, out_file, pickle.HIGHEST_PROTOCOL)


def load_network(filename):
    with open(filename, 'rb') as in_file:
        return pickle.load(in_file)


"""
Main
"""
def main():
    ann = Network([28*28, 30, 10])
    training_data, test_data = mnist.load_data()

    # reformat mnist data into tuples of (input, expected output)
    # flatten images into "1-D arrays" (mxn but n = 1)
    # and scale image down to the range [0,1]
    training_data = [(x.reshape(28*28, 1) / 255, y) for x, y in zip(training_data[0], training_data[1])]
    test_data = [(x.reshape(28*28, 1) / 255, y) for x, y in zip(test_data[0], test_data[1])]

    ann.SGD(training_data, 30, 10, 3e-1, test_data)
    save_network(ann, "test.pkl")


if __name__ == "__main__":
    main()

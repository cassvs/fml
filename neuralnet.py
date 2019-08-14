"""
FML: a Farcical Machine Learning library.
Cass Smith, August 2019

Inspired by Grant Sanderson's excellent videos on neural networks.

Classes:
    Functions:      Math for calculating activations and errors.
    Layer:    A layer with activations, weights, and biases.
    NeuralNet:      A collection of layers.
"""

from fml.matrix import *
import math
import random
import json


class Functions():
    """
    Activation and error functions.

    Functions:
        sigmoid:                Sigmoid (logistic) activation function.
        sigmoidDerivative:      Partial derivative of sigmoid function.
        relu:                   Rectified linear unit (ReLU) function.
        reluDerivative:         Partial derivative of ReLU function.
        identity:               The identity function.
        identityDerivative:     1.
        euclidean:              Euclidean distance error function.
        euclideanDerivative:    Partial derivative of Euclidean function.
    """

    @staticmethod
    def sigmoid(number):
        """Calculate the sigmoid (aka logistic) activation function of an input."""
        # Don't use this.
        # Backpropagating with this transfer function causes the weights and biases
        # in each successive layer to become exponentially larger, leading to a
        # floating point overflow.
        return 1 / (1 + math.e ** (-(number)))

    @staticmethod
    def sigmoidDerivative(number):
        """Calculate the partial derivative of the sigmoid activation function."""
        return number * (1 - number)

    @staticmethod
    def relu(number):
        """Return input if input is greater than zero, else return zero."""
        # Don't use this.
        # Backpropagating with this transfer function causes random neurons to get
        # stuck at zero, with no way of recovering.
        return number if number > 0 else 0

    @staticmethod
    def reluDerivative(number):
        """Return 1 if input is greater than zero, else return zero."""
        return 1 if number > 0 else 0

    @staticmethod
    def identity(number):
        """Return input."""
        # This is the only transfer function that even comes close to working.
        # It's still terrible though. Don't use it.
        return number

    @staticmethod
    def identityDerivative(number):
        """Return 1."""
        return 1

    @staticmethod
    def euclidean(x, y):
        """Calculate the error between two values."""
        return 0.5 * math.pow(x - y, 2)

    @staticmethod
    def euclideanDerivative(x, y):
        """Calculate the partial derivative of the Euclidean error function"""
        return x - y


class Layer(object):
    """Layer with activations, weights, and biases."""

    def __init__(self, nodes, prevNodes, transfer, dtransfer):
        """
        Construct a Layer.

        Parameters:
            nodes:      How many nodes this layer has.
            prevNodes:  How many nodes the previous layer has.
            transfer:   The activation function to use for this layer.
        """
        self.activations = Matrix(nodes, 1)
        self.biases = Matrix(nodes, 1)
        self.weights = Matrix(nodes, prevNodes)
        self.transfer = transfer
        self.dtransfer = dtransfer

    def activate(self, source):
        """
        Calculate activations for this layer.

        Parameters:
            source: List of activations in the previous layer.
        """
        self.activations = (self.weights * source +
                            self.biases).map(self.transfer)

    def backprop(self, dactivations, prevActivations, rate):
        """
        Tweak the weights and biases of this layer.

        Parameters:
            dactivations:       Matrix containing the partial derivatives of the activations of this layer.
            prevActivations:    Matrix containing the activations of the previous layer.
            rate:               Learning rate.
        """
        # This is the source of all the bugs, I'm pretty sure.
        dtransfer = (self.weights * prevActivations +
                     self.biases).map(lambda cell: self.dtransfer(cell))
        dbiases = Matrix(*self.biases.dimensions)
        for row in range(len(dbiases.rows)):
            dbiases[(row, 0)] = dtransfer[(row, 0)] * dactivations[(row, 0)]
        dweights = prevActivations * Matrix(dbiases.cols)
        dprevActivations = Matrix(self.weights.cols) * dbiases
        for row in range(len(self.biases.rows)):
            self.biases[(row, 0)] -= rate * dbiases[(row, 0)]
        dweights = Matrix(dweights.cols)
        for row in range(len(self.weights.rows)):
            for col in range(len(self.weights.cols)):
                self.weights[(row, col)] -= rate * dweights[(row, col)]
        return dprevActivations


class NeuralNet(object):
    """Collection of layers forming a network."""

    def __init__(self, layers, transfer, error):
        """
        Create new NeuralNet.

        Parameters:
            layers:     List containing numbers of nodes in each layer.
            transfer:   Activation function for this network.
            error:      Error function for this network.
        """
        if error == "euclidean":
            self.error = Functions.euclidean
            self.derror = Functions.euclideanDerivative
        else:
            self.error = Functions.euclidean
            self.derror = Functions.euclideanDerivative

        if transfer == "sigmoid":
            self.transfer = Functions.sigmoid
            self.dtransfer = Functions.sigmoidDerivative
        elif transfer == "relu":
            self.transfer = Functions.relu
            self.dtransfer = Functions.reluDerivative
        elif transfer == "identity":
            self.transfer = Functions.identity
            self.dtransfer = Functions.identityDerivative
        else:
            self.transfer = Functions.sigmoid
            self.dtransfer = Functions.sigmoidDerivative

        self.layers = [Layer(layers[i], layers[i - 1] if i else 0,
                             self.transfer, self.dtransfer) for i in range(len(layers))]

    def predict(self, input):
        """
        Forward-propagate activations through the network, return activations of the output layer.

        Parameters:
            input: Values to activate the input layer.
        """
        inputIterator = iter(input)
        self.layers[0].activations.mapInPlace(lambda cell: next(inputIterator))
        [self.layers[i].activate(self.layers[i - 1].activations)
         for i in range(1, len(self.layers))]
        return self.layers[len(self.layers) - 1].activations

    def randomize(self, lower, upper):
        """
        Set weights and biases in all (non-input) layers to random values.

        Parameters:
            lower:  Lower bound for random values.
            upper:  Upper bound for random values.
        """
        for i in range(1, len(self.layers)):
            self.layers[i].weights.mapInPlace(
                lambda cell: random.uniform(lower, upper))
            self.layers[i].biases.mapInPlace(
                lambda cell: random.uniform(lower, upper))

    def save(self):
        """Return a JSON string representing this network."""
        dumpable = {
            "layers": [len(l.activations.rows) for l in self.layers],
            "weights": [l.weights.rows for l in self.layers],
            "biases": [l.biases.rows for l in self.layers],
            "transfer": self.transfer.__name__,
            "error": self.error.__name__
        }
        return json.dumps(dumpable, indent=4)

    @staticmethod
    def load(str):
        """Convert a JSON-encoded network into a new NeuralNet object."""
        l = json.loads(str)
        newnn = NeuralNet(l["layers"], getattr(
            Functions, l["transfer"]), getattr(Functions, l["error"]))
        for la, lw, lb in zip(newnn.layers, l["weights"], l["biases"]):
            la.weights = Matrix(lw)
            la.biases = Matrix(lb)
        return newnn

    def train(self, dataIn, expectedOut, rounds, rate):
        """
        Learn from labelled data.

        Parameters:
            dataIn:         List of lists of input values.
            expectedOut:    Labels for input data.
            rounds:         Number of times to run through the dataset.
            rate:           Learning rate.

        Currently sort of working.
        """
        mappedData = list(zip(dataIn, expectedOut))
        for r in range(rounds):
            random.shuffle(mappedData)
            for d, e in mappedData:
                self.backpropagate(d, e, rate)

    def backpropagate(self, dataIn, expectedOut, rate):
        """
        Tweak the weights and biases of each layer.

        Parameters:
            dataIn:         Flat list containing one training example's input data.
            expectedOut:    Flat list containing the expected output for the training example.
            rate:           Learning rate.
        """
        actualOutput = self.predict(dataIn)
        error = Matrix(*actualOutput.dimensions)
        derror = Matrix(*actualOutput.dimensions)

        for row in range(len(actualOutput.rows)):
            error[(row, 0)] = self.error(
                actualOutput[(row, 0)], expectedOut[row])
            derror[(row, 0)] = self.derror(
                actualOutput[(row, 0)], expectedOut[row])

        # Pass the activation derivative generated by the previous backprop step to the next backprop step.
        lastderror = derror
        for l in range(len(self.layers) - 1, 0, -1):
            lastderror = self.layers[l].backprop(
                lastderror, self.layers[l - 1].activations, rate)

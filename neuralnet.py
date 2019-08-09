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
        sigmoid:    Sigmoid (logistic) activation function.
        diffSquares:  Difference of squared outputs error function.
    """

    @staticmethod
    def sigmoid(number):
        """Calculate the sigmoid (aka logistic) activation function of an input."""
        return 1 / (1 + math.exp(-number))

    @staticmethod
    def diffSquares(a, b):
        """Calculate the difference between two lists."""
        return sum([x * x - y * y for x, y in zip(a, b)])


class Layer(object):
    """Layer with activations, weights, and biases."""

    def __init__(self, nodes, prevNodes, transfer):
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

    def activate(self, source):
        """
        Calculate activations for this layer.

        Parameters:
            source: List of activations in the previous layer.
        """
        self.activations = (self.weights * source +
                            self.biases).map(self.transfer)


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
        self.transfer = transfer
        self.error = error
        self.layers = [Layer(layers[i], layers[i - 1] if i else 0,
                             self.transfer) for i in range(len(layers))]

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
        prediction = []
        self.layers[len(self.layers) - 1
                    ].activations.map(lambda cell: prediction.append(cell))
        return prediction

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

    def train(self, dataIn, expectedOut, rounds):
        """
        Learn from labelled data.

        Currently not working.
        """
        mappedData = zip(dataIn, expectedOut)
        cost = 0
        for d, e in random.shuffle(mappedData):
            testPrediction = self.predict(d)
            cost += self.error(testPrediction, e)
        cost /= len(dataIn)
        # TODO: Gradient descent stuffs

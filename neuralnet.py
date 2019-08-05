"""
FML: a Farcical Machine Learning library.
Cass Smith, August 2019

Inspired by Grant Sanderson's excellent videos on neural networks.

Classes:
    Functions:      Math for calculating activations and errors.
    InputLayer:     A layer with only activations.
    HiddenLayer:    A layer with activations, weights, and biases.
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
        euclidean:  Euclidean distance error function.
    """

    @staticmethod
    def sigmoid(number):
        """Calculate the sigmoid (aka logistic) activation function of an input."""
        return 1 / (1 + math.exp(-number))

    @staticmethod
    def euclidean(a, b):
        """Calculate the Euclidean distance between two vectors."""
        ## TODO: That's not even close to what it is
        return sum([x * x - y * y for x, y in zip(a, b)])


class InputLayer(object):

    """Layer with nothing but activations."""
    def __init__(self, nodes):
        self.activations = Matrix(nodes, 1)


class HiddenLayer(InputLayer):
    """Layer with activations, weights, and biases."""

    def __init__(self, nodes, prevNodes, transfer):
        """
        Construct a HiddenLayer.

        Parameters:
            nodes:      How many nodes this layer has.
            prevNodes:  How many nodes the previous layer has.
            transfer:   The activation function to use for this layer.
        """
        super().__init__(nodes)
        self.biases = Matrix(nodes, 1)
        self.weights = Matrix(nodes, prevNodes)
        self.transfer = transfer

    def activate(self, source):
        """
        Calculate activations for this layer.

        Parameters:
            source: List of activations in the previous layer.
        """
        self.activations = self.weights.multiply(
            source).add(self.biases)
        self.activations.forEach(lambda cell, row, col: self.activations.setCell(
            row, col, self.transfer(cell)))


class NeuralNet(object):
    """Collection of layers forming a network."""

    def __init__(self, input, hidden, output, transfer, error):
        """
        Create new NeuralNet.

        Parameters:
            input:      Number of nodes in the input layer.
            hidden:     List containing numbers of nodes in each hidden layer.
            output:     Number of nodes in output layer.
            transfer:   Activation function for this network.
            error:      Error function for this network.
        """
        self.transfer = transfer
        self.error = error
        self.input = InputLayer(input)
        self.hidden = [HiddenLayer(
            hidden[i], hidden[i - 1] if i else input, self.transfer) for i in range(len(hidden))]
        self.output = HiddenLayer(
            output, hidden[len(hidden) - 1], self.transfer)

    def predict(self, input):
        """
        Forward-propagate activations through the network, return activations of the output layer.

        Parameters:
            input: Values to activate the input layer.
        """
        self.input.activations.forEach(
            lambda cell, row, col: self.input.activations.setCell(row, col, input[col]))
        [self.hidden[i].activate(self.hidden[i - 1].activations if i else self.input.activations)
         for i in range(len(self.hidden))]
        self.output.activate(self.hidden[len(self.hidden) - 1].activations)
        prediction = []
        self.output.activations.forEach(
            lambda cell, row, col: prediction.append(cell))
        return prediction

    def randomize(self, lower, upper):
        """
        Set weights and biases in all (non-input) layers to random values.

        Parameters:
            lower:  Lower bound for random values.
            upper:  Upper bound for random values.
        """
        for h in self.hidden:
            h.weights.forEach(lambda w, r, c: h.weights.setCell(
                r, c, random.uniform(lower, upper)))
            h.biases.forEach(lambda w, r, c: h.biases.setCell(
                r, c, random.uniform(lower, upper)))
        self.output.weights.forEach(
            lambda w, r, c: self.output.weights.setCell(r, c, random.uniform(lower, upper)))
        self.output.biases.forEach(
            lambda w, r, c: self.output.biases.setCell(r, c, random.uniform(lower, upper)))

    def save(self):
        """Return a JSON string representing this network."""
        dumpable = {
            "layers": {
                "input": self.input.activations.cells(),
                "hidden": [h.activations.cells() for h in self.hidden],
                "output": self.output.activations.cells()
            },
            "weights": {
                "hidden": [h.weights.data for h in self.hidden],
                "output": self.output.weights.data
            },
            "biases": {
                "hidden": [h.biases.data for h in self.hidden],
                "output": self.output.biases.data
            },
            "transfer": self.transfer.__name__,
            "error": self.error.__name__
        }
        return json.dumps(dumpable, indent=4)

    @staticmethod
    def load(str):
        """Convert a JSON-encoded network into a new NeuralNet object."""
        l = json.loads(str)
        newnn = NeuralNet(l["layers"]["input"], l["layers"]["hidden"], l["layers"]["output"], getattr(
            Functions, l["transfer"]), getattr(Functions, l["error"]))
        for h, hw, hb in zip(newnn.hidden, l["weights"]["hidden"], l["biases"]["hidden"]):
            h.weights.data = hw
            h.biases.data = hb
        newnn.output.weights.data = l["weights"]["output"]
        newnn.output.biases.data = l["biases"]["output"]
        return newnn

    def train(self, dataIn, expectedOut, rounds):
        """
        Learn from labelled data.
        """
        mappedData = zip(dataIn, expectedOut)
        cost = 0
        for d, e in random.shuffle(mappedData):
            testPrediction = self.predict(d)
            cost += self.error(testPrediction, e)
        cost /= len(dataIn)
            # TODO: Gradient descent stuffs

import sys
import random
sys.path.append("../..")

import fml.neuralnet as nn
import fml.matrix as m


def main():

    tests = 0
    passed = 0

    tests += 1
    print("Random garbage test:")
    print("Creating and initializing a new net with random weights and biases")
    mynn = nn.NeuralNet([4, 3, 2], "sigmoid", "euclidean")
    mynn.randomize(-10, 10)
    print("Creating two predictions with different input data to see if the results are different")
    prediction1 = mynn.predict([1, 1, 1, 1])
    prediction2 = mynn.predict([0, 0, 0, 0])
    print([1, 1, 1, 1], "->", prediction1)
    print([0, 0, 0, 0], "->", prediction2)
    if prediction1 != prediction2:
        print("Success!\n")
        passed += 1
    else :
        print("Test failed!\n")

    tests += 1
    print("Random garbage consistency test:")
    print("Does the net produce the same output twice from the same input?")
    prediction3 = mynn.predict([4, 4, 4, 4])
    prediction4 = mynn.predict([4, 4, 4, 4])
    print(prediction3)
    print(prediction4)
    if prediction3 == prediction4:
        print("Success!\n")
        passed += 1
    else:
        print("Test failed!\n")

    tests += 1
    print("Save/load test:")
    print("Does a net that has been restored from a JSON save behave the same as one that hasn't?")
    savestr = mynn.save()
    #print(savestr)
    nncopy = nn.NeuralNet.load(savestr)
    prediction5 = nncopy.predict([4, 4, 4, 4])
    print(prediction4, "<- Original net")
    print(prediction5, "<- Saved and restored net")
    if prediction5 == prediction4:
        print("Success!\n")
        passed += 1
    else:
        print("Test failed!\n")

    tests += 1
    print("Training test:")
    print("Can a net learn to be an OR gate?")
    trainingInputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    trainingOutputs = [[0, 1], [1, 0], [1, 0], [1, 0]]
    ornn = nn.NeuralNet([2, 2, 2], "identity", "euclidean")
    ornn.randomize(-1, 1)
    ornn.train(trainingInputs, trainingOutputs, 500, 0.1)
    producedOutput = []
    for i in range(len(trainingInputs)):
        producedOutput.append(ornn.predict(trainingInputs[i]).cols[0])
    producedOutput = m.Matrix(*producedOutput)
    expectedOutput = m.Matrix(trainingOutputs)
    for i in range(len(producedOutput.rows)):
        print(trainingInputs[i], "->", producedOutput.rows[i])
    producedOutput.mapInPlace(lambda cell: round(cell))
    if producedOutput == expectedOutput:
        passed += 1
        print("Success!\n")
    else:
        print("Test failed!\n")


    print("Tests passed: ", passed, "/", tests)
main()

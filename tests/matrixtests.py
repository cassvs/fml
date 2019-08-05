import sys
import random
sys.path.append("../..")

#from fml.matrix import Matrix
import fml.matrix as mat


def main():

    tests = 1
    passed = 0

    print("Identity matrix multiplication test:")
    a = mat.Matrix(4, 4)
    i = mat.Matrix(4, 4)
    a.forEach(lambda cell, row, col: a.setCell(row, col, random.randint(0, 10)))
    i.forEach(lambda cell, row, col: i.setCell(row, col, 1 if row == col else 0))
    print("Multiplying\n", a.data, "\nby\n", i.data)
    print("The result should be the same as the first matrix.")
    r = a.multiply(i)
    print(r.data)
    if a.data == r.data:
        print("Success!")
        passed += 1
    else :
        print("Test failed!")

    print("Tests passed: ", passed, "/", tests)

main()

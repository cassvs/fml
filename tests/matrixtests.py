import sys
import random
sys.path.append("../..")
import fml.matrix as mat

#from fml.matrix import Matrix


def main():

    tests = 0
    passed = 0

    tests += 1
    print("Identity matrix multiplication test:")
    a = mat.Matrix(4, 4)
    i = mat.Matrix([[1, 0, 0, 0], [0, 1, 0, 0],
                    [0, 0, 1, 0], [0, 0, 0, 1]])
    for r in range(len(a.rows)):
        for c in range(len(a.cols)):
            a[(r, c)] = random.randint(0, 10)
        # print(cell)
    print("Multiplying\n", a, "\nby\n", i)
    print("The result should be the same as the first matrix.")
    r = a * i
    print(r)
    if a == r:
        print("Success!\n")
        passed += 1
    else:
        print("Test failed!\n")

    tests += 1
    print("Matrix-scalar multiplication test:")
    b = mat.Matrix([[1, 1, 1, 1], [1, 1, 1, 1],
                    [1, 1, 1, 1], [1, 1, 1, 1]])
    scalar = 2.2
    print("Multiplying\n", b, "\nby\n", scalar)
    s = b * scalar
    print(s)
    if s == mat.Matrix([[2.2, 2.2, 2.2, 2.2],
                        [2.2, 2.2, 2.2, 2.2],
                        [2.2, 2.2, 2.2, 2.2],
                        [2.2, 2.2, 2.2, 2.2]]):
        print("Success!\n")
        passed += 1
    else:
        print("Test failed!\n")

    tests += 1
    print("Addition test:")
    print("Adding\n", b, "\nto\n", s)
    c = b + s
    print(c)
    if c == mat.Matrix([[3.2, 3.2, 3.2, 3.2],
                        [3.2, 3.2, 3.2, 3.2],
                        [3.2, 3.2, 3.2, 3.2],
                        [3.2, 3.2, 3.2, 3.2]]):
        print("Success!\n")
        passed += 1
    else:
        print("Test failed!\n")

    print("Tests passed: ", passed, "/", tests)


main()

"""
Matrix module for FML.
Cass Smith, August 2019
"""

class Matrix(object):
    """Basically a list of lists, with some math methods."""

    def __init__(self, *args):
        """
        Create a new Matrix object.

        Parameters:
            If passed a list of two integers, create a new matrix with those dimensions.
            If passed several lists, or a list of lists, copy them into a new matrix.
        """
        # "Python forces you to write readable code"
        # And Other Hilarious Jokes You Can Tell Yourself
        if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
            self._data = [[0 for x in range(args[1])] for y in range(args[0])]
        elif isinstance(args[0], list):
            lol = args
            if isinstance(args[0][0], list):
                lol = args[0]
            self._data = [[0 for x in range(len(lol[0]))] for y in range(len(lol))]
            for r in range(len(self.rows)):
                for c in range(len(self.cols)):
                    self[(r, c)] = lol[r][c]
        else:
            raise TypeError("Can't create matrix")

    def map(self, func):
        """
        Apply a function to each cell in the matrix, returning a new matrix containing the result.

        Parameters:
            func:  Function to apply. Must accept one parameter and return a value.
        """
        result = Matrix(len(self.rows), len(self.cols))
        for r in range(len(self.rows)):
            for c in range(len(self.cols)):
                result[(r, c)] = func(self[(r, c)])
        return result

    def mapInPlace(self, func):
        """Same as `map`, but apply the result to this matrix instead of returning a new one."""
        for r in range(len(self.rows)):
            for c in range(len(self.cols)):
                self[(r, c)] = func(self[(r, c)])
        return self

    @property
    def rows(self):
        """Return list of rows."""
        return self._data

    @property
    def cols(self):
        """
        Return list of columns.

        Equivalent to reflecting the matrix about its diagonal.
        """
        return [list(z) for z in zip(*self._data)]

    @property
    def dimensions(self):
        """Return a tuple describing the size of the matrix."""
        return (len(self.rows), len(self.cols))

    def __getitem__(self, key):
        """Access cell by location ([row, column])."""
        return self._data[key[0]][key[1]]

    def __setitem__(self, key, value):
        """Assign value to cell at location ([row, column])."""
        self._data[key[0]][key[1]] = value

    def __mul__(self, other):
        """
        Return a new matrix containing the product of matrix-matrix or matrix-scalar multiplication.

        Parameters:
            other: Matrix or scalar.
        """
        # My native language is C. Is it too obvious? ;)
        if isinstance(other, Matrix) and len(self.cols) == len(other.rows):
            product = Matrix(len(self.rows), len(other.cols))
            for i in range(len(product.rows)):
                for j in range(len(product.cols)):
                    for k in range(len(other.rows)):
                        product[(i, j)] += self[(i, k)] * other[(k, j)]
            return product
        elif isinstance(other, int) or isinstance(other, float):
            return Matrix([[cell * other for cell in row] for row in self.rows])
        else:
            raise TypeError('Cannot multiply Matrix and incompatible type.')

    def __add__(self, other):
        """
        Return a new matrix containing the sum of two matrices.

        Parameters:
            other: Another matrix of the same dimensions as this one.
        """
        if isinstance(other, Matrix) and self.dimensions == other.dimensions:
            # This is somewhat more "Pythonic" than the multiply routine, I suppose
            return Matrix([[selfCell + otherCell for selfCell, otherCell in zip(selfRow, otherRow)] for selfRow, otherRow in zip(self.rows, other.rows)])
        else:
            raise TypeError('Cannot add Matrix and incompatible type.')

    def __iter__(self):
        """Return a MatrixIterator object for this matrix."""
        return MatrixIterator(self)

    def __len__(self):
        """Return the number of cells in the Matrix."""
        return len(self.rows) * len(self.cols)

    def __repr__(self):
        """Return a printable representation of this matrix."""
        return self._data.__repr__()

    def __eq__(self, other):
        """Return true if self and other contain the same data."""
        if isinstance(other, Matrix) and self.dimensions == other.dimensions:
            for selfCell, otherCell in zip(self, other):
                if selfCell != otherCell:
                    return False
            return True
        else:
            return False


class MatrixIterator(object):
    """
    Iterator object for traversing a matrix as if 'twer a flat list.
    """
    # Almost completely useless, might delete later
    def __init__(self, matrix):
        self.matrix = matrix
        self.r = 0
        self.c = 0

    def __next__(self):
        oldr = self.r
        oldc = self.c
        self.c += 1
        if self.c >= len(self.matrix.cols):
            self.c = 0
            self.r += 1
        if self.r >= len(self.matrix.rows):
            raise StopIteration()
        return self.matrix[(oldr, oldc)]

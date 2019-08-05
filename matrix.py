"""
Matrix module for FML.
Cass Smith, August 2019
"""

class Matrix(object):
    """Basically a list of lists, with some math methods."""

    def __init__(self, rows, cols):
        """
        Create a new Matrix object.

        Parameters:
            rows: The number of rows (the height) of the new Matrix.
            cols: The number of columns (the width) of the new Matrix.
        """
        self.rows = rows
        self.cols = cols
        self.data = []

        for i in range(self.rows):
            self.data.append([])
            for j in range(self.cols):
                self.data[i].append(0)
        ## TODO: yucky

    def cells(self):
        """Return the number of cells in the Matrix."""
        return self.rows * self.cols

    def setCell(self, row, col, val):
        """
        Set the value of a particular cell.

        Parameters:
            row:    The row of the cell.
            col:    The column of the cell.
            val:    Value to store in the cell.
        """
        self.data[row][col] = val

    def forEach(self, func):
        """Apply a function to each cell."""
        for i in range(self.rows):
            for j in range(self.cols):
                func(self.data[i][j], i, j)
        return self

    def multiply(self, other):
        """
        Multiply this Matrix with another Matrix and return a new Matrix containing the result.
        """
        if (self.cols != other.rows):
            raise TypeError('Cannot multiply matrices')
            return

        product = Matrix(self.rows, other.cols)
        for i in range(product.rows):
            for j in range(product.cols):
                for k in range(other.rows):
                    product.data[i][j] += self.data[i][k] * other.data[k][j]
        return product

    def add(self, other):
        """
        Add this Matrix to another Matrix and return a new Matrix containing the result.
        """
        if (self.rows != other.rows or self.cols != other.cols):
            raise TypeError('Cannot add matrices with different dimensions')
            return

        sum = Matrix(self.rows, self.cols)
        for i in range(sum.rows):
            for j in range(sum.cols):
                sum.data[i][j] = self.data[i][j] + other.data[i][j]
        return sum

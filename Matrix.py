import random
import Logger

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.matrix = [[0] * self.cols for i in range(self.rows)]

    def randomise(self):
        for x in range(self.rows):
            for y in range(self.cols):
                self.matrix[x][y] = random.random() * 2 - 1

    @staticmethod
    def fromArray(arr):
        m = Matrix(len(arr), 1)
        for x in range(0, len(arr)):
            m.matrix[x][0] = arr[x]
        return m

    def toArray(self):
        arr = []
        for x in range(0, self.rows):
            for y in range(0, self.cols):
                arr.append(self.matrix[x][y])
        return arr

    def add(self, n):
        if (isinstance(n, Matrix)):
            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    self.matrix[x][y] += n.matrix[x][y]
        else:
            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    self.matrix[x][y] += n

    @staticmethod
    def subtract(m1, m2):
        result = Matrix(m1.rows, m1.cols)
        for x in range(0, result.rows):
            for y in range(0, result.cols):
                result.matrix[x][y] = m1.matrix[x][y] - m2.matrix[x][y]
                
        return result

    @staticmethod
    def multiply(m1, m2):
        if (m1.cols != m2.rows):
            Logger.Log("Columns of m1 must match rows of m2.", "ERROR")
            return None
        
        result = Matrix(m1.rows, m2.cols)
        for x in range(result.rows):
            for y in range(result.cols):
                total = 0
                for k in range(m1.cols):
                   total += m1.matrix[x][k] * m2.matrix[k][y]
                result.matrix[x][y] = total
        return result

    def mult(self, n):
        if (isinstance(n, Matrix)):
            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    self.matrix[x][y] *= n.matrix[x][y]
        else:
            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    self.matrix[x][y] *= n
            
    def map(self, func):
        for x in range(0, self.rows):
            for y in range(0, self.cols):
                val = self.matrix[x][y]
                self.matrix[x][y] = func(val)

    @staticmethod
    def mapMatrix(m, func):
        result = Matrix(m.rows, m.cols)
        for x in range(0, m.rows):
            for y in range(0, m.cols):
                val = m.matrix[x][y]
                result.matrix[x][y] = func(val)
        return result

    @staticmethod
    def transpose(m):
        result = Matrix(m.cols, m.rows)
        for x in range(0, m.rows):
            for y in range(m.cols):
                result.matrix[y][x] = m.matrix[x][y]

        return result

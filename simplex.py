import numpy as np


np.set_printoptions(suppress=True)


class SimplexMethod:
    """
        Simplex method algorithm that works with standard/canonical form: max(c.T*x) s.t Ax<=b and x>=0.

        Attributes:
            c: coefficients of the objective function
            a: coefficients of the inequalities
            b: right-hand side of the inequalities

    """
    def __init__(self, c: np.ndarray, a: np.ndarray, b: np.ndarray): #maximise by default
        self.c = c
        self.a = a
        self.b = b
    def initialize_table(self):

        c = self.c
        a = self.a
        b = self.b
        piece1 = np.concatenate((a, c * (-1)), axis=0)
        piece2 = np.concatenate((np.identity(len(a)), np.zeros(shape=(1, len(a)))), axis=0)
        piece3 = np.concatenate((np.zeros(shape=(1,len(a))), np.array([[1]])), axis=1).T
        piece4 = np.concatenate((b, np.array([0])), axis=0)
        piece4 = piece4.reshape((len(piece4), 1))
        table = np.concatenate((piece1, piece2, piece3, piece4), axis=1)

        return table

    @staticmethod
    def identify_pivot(table):

        pivot_column_index = np.argmin(table[-1])
        pivot_column = table[:,pivot_column_index][:-1]
        pivots = np.divide(table[:, -1][:-1], pivot_column)
        pivot_row_index = np.argmin(pivots)
        return pivot_row_index, pivot_column_index

    @staticmethod
    def pivot_gauss_elimination(table, pivot_row, pivot_col):
        pivot_value = table[pivot_row][pivot_col]

        # Make pivot element 1
        table[pivot_row] = table[pivot_row] / pivot_value

        # Make sure pivot column other elements are 0
        rows = [i for i in range(table.shape[0]) if i != pivot_row]
        for row in rows:
            table[row] = table[row] - table[row][pivot_col] * table[pivot_row]

        return table

    def perform_simplex(self):
        table = self.initialize_table()

        ##keep track of current variables
        basis = [f's{i + 1}' for i in range(len(table[:-1]))]

        i = 0
        while not np.all(table[-1] >= 0):
            smallest_pivot_row_index, pivot_column_index = self.identify_pivot(table)

            basis[smallest_pivot_row_index] = f'x{pivot_column_index + 1}'

            table_updated = self.pivot_gauss_elimination(table, smallest_pivot_row_index, pivot_column_index)
            table = table_updated
            i += 1
        return table[:,-1], basis, i


"REMARK"
#s1, s2... stand for slack variables(not important)
#If there are e.g. variables x1 and x3, but no x2, this means that 0 should be assigned to x2

def perform_simplex(c, a, b):
    table, variables, iters = SimplexMethod(c, a, b).perform_simplex()
    parameters = table[:-1]
    value = table[-1]
    result = ('----------------------\n'
              ''f'{": ".join([f"{variables[i]}: {parameters[i]}" for i in range(len(variables))])} \n'
              f'The optimal value f(x1, x2...) is {value} \n'
              f'{iters} iterations were performed \n'
              f'---------------------')
    return result




"2 variables"

problem1 = perform_simplex(c = np.array([[30, 40]]),
                      a = np.array([[2, 1],
                                    [1, 1],
                                    [1, 2]]),
                      b = np.array([10, 7, 12]))


problem2 = perform_simplex(c = np.array([[7, 8]]),
                           a = np.array([[2, 3],
                                         [1, 1]]),
                           b = np.array([1000, 800]))

problem3 = perform_simplex(c = np.array([[5, 4]]),
                         a = np.array([[1, 2],
                                       [3, 2]]),
                         b = np.array([20, 30]))

problem4 = perform_simplex(c = np.array([[5, 2]]),
                         a = np.array([[3, 2],
                                       [1, 1],
                                       [1, 3],
                                       [1, 1]]),
                         b = np.array([60, 26, 45, 90]))


problem5 = perform_simplex(c = np.array([[70, 90]]),
                         a = np.array([[40, 30],
                                       [30, 50]]),
                         b = np.array([1800, 2000]))

"10 variables"

problem6 = perform_simplex(c = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
                           a = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]]),
                           b = np.array([55, 150]))

problem7 = perform_simplex(c = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 1]]),
                           a = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]]),
                           b = np.array([600, 90]))

problem8 = perform_simplex(c = np.array([[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]]),
                           a = np.array([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                         [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
                                         [50, 60, 70, 80, 90, 100, 10, 20, 30, 40],
                                         [40, 30, 20, 10, 100, 90, 80, 70, 60, 50]]),
                           b = np.array([5500, 100, 6000, 5000]))

problem9 = perform_simplex(c = np.array([[200, 300, 400, 500, 600, 700, 800, 900, 1000, 100]]),
                           a = np.array([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                         [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
                                         [50, 70, 90, 110, 130, 150, 170, 190, 210, 230],
                                         [230, 210, 190, 170, 150, 130, 110, 90, 70, 50]]),
                           b = np.array([6000, 6000, 13000, 13000]))

problem10 = perform_simplex(c = np.array([[210, 320, 430, 540, 650, 760, 870, 980, 1090, 1200]]),
                           a = np.array([[20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                                         [110, 100, 90, 80, 70, 60, 50, 40, 30, 20],
                                         [60, 80, 100, 120, 140, 160, 180, 200, 220, 240],
                                         [240, 220, 200, 180, 160, 140, 120, 100, 80, 60],
                                         [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
                                         [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]),
                           b = np.array([6200, 6200, 13500, 13500, 6000, 6000]))
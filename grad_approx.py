import numpy as np


class Function:
    def __init__(self, func):
        self.func = func
    def calc_f(self, x, y):
        return self.func(x, y)

    def estimate_grad(self, x, y, h = np.sqrt(1.1 * 10 ** -16)):

        #central difference formula:

        #first partial derivative
        df_dx_f = lambda x_val, y_val: (self.calc_f(x_val + h, y_val) - self.calc_f(x_val - h, y_val)) / (2 * h)
        df_dx = df_dx_f(x, y)

        #second partial derivative
        df_dy_f = lambda x_val, y_val: (self.calc_f(x_val, y_val + h) - self.calc_f(x_val, y_val - h)) / (2 * h)
        df_dy = df_dy_f(x, y)

        grad = np.array([df_dx, df_dy])

        return grad


f1 = lambda x, y : 100 * (y - x ** 2) ** 2 + (1 - x) ** 2
f2 = lambda x, y: 150 * (x * y) ** 2 + (0.5 * x + 2 * y - 2) ** 2

f = Function(f1)
print(f.estimate_grad(1, 1))






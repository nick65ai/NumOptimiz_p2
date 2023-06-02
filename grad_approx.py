import numpy as np

#increment value
u = np.sqrt(1.1 * 10 ** -16)

class Function:
    def __init__(self, func):
        self.func = func

    def f(self, x, y):
        return self.func(x, y)

    def estimate_grad(self, x, y, h = u):
        # central difference formula:

        #first partial derivative
        df_dx = (self.f(x + h, y) - self.f(x - h, y)) / (2 * h)
        #second partial derivative
        df_dy = (self.f(x, y + h) - self.f(x, y - h)) / (2 * h)

        grad = np.array([df_dx, df_dy])

        return grad

    def estimate_hessian(self, x, y, h = 0.1):

        hessian = np.zeros((2, 2))

        #finite difference formula

        df_dx_2 = (self.f(x + h, y) - 2 * self.f(x, y) + self.f(x - h, y)) / (h ** 2)
        df_dy_2 = (self.f(x, y + h) - 2 * self.f(x, y) + self.f(x, y - h)) / (h ** 2)
        df_2_symmetric = (self.f(x + h, y + h) - self.f(x + h, y - h) - self.f(x - h, y + h) + self.f(x - h, y - h)) / (4 * h ** 2)

        hessian[0, 0] = df_dx_2
        hessian[1, 1] = df_dy_2
        hessian[0, 1] = hessian[1, 0] = df_2_symmetric

        return hessian


f1 = lambda x, y : 100 * (y - x ** 2) ** 2 + (1 - x) ** 2
f2 = lambda x, y: 150 * (x * y) ** 2 + (0.5 * x + 2 * y - 2) ** 2

f = Function(f1)
print(f.estimate_grad(100, 100))
print(f.estimate_hessian(-67, -2))





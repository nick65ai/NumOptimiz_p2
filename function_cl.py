import numpy as np
import sympy as sp
from numpy import linalg
from sympy import symbols, lambdify

x, y = symbols('x y')


class my_func:
    def __init__(self, function):
        self.function = function
        f = lambdify([x, y], function)
        self.f = f

    def calc_f_at_point(self, val):
        return self.f(val[0], val[1])

    def calc_grad_at_point(self, val):
        grad_m = sp.Matrix([sp.diff(self.function, var) for var in [x, y]])
        grad = grad_m.subs({x: val[0], y: val[1]})
        grad = np.array([float(grad[i].evalf()) for i in range(len(grad))], dtype=np.float64)
        return np.array(grad, dtype=np.float64)


def backtracking(func, point, alpha, c, p):
    p_k = - np.linalg.inv(func.hessian_at_p(point)) @ func.calc_grad_at_point(point)

    # algorithm 3.1 from the book

    while True:
        term1 = func.calc_f_at_point(val=point + alpha * p_k)
        term2 = func.calc_f_at_point(val=point) + c * alpha * func.calc_grad_at_point(val=point).T @ p_k
        if linalg.norm(term1) <= linalg.norm(term2):
            break
        alpha *= p

    return alpha
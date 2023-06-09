import numpy as np

def fletcher_reeves(f, grad_f, x0, epsilon=1e-6, max_iter=1000):
    x = x0
    c = 0.5
    alpha = 0.5
    grad = grad_f(x)
    p = -grad
    k = 0

    while np.linalg.norm(grad) > epsilon and k < max_iter:
        step_size = backtracking_line_search(f, grad_f, x, p, alpha, c)

        x_new = x + step_size * p
        grad_new = grad_f(x_new)

        beta_fr = np.dot(grad_new, grad_new) / np.dot(grad, grad)
        p = -grad_new + beta_fr * p

        x = x_new
        grad = grad_new
        k += 1

    return x


def backtracking_line_search(f, grad_f, x, p, alpha, c):
    while f(x + alpha * p) > f(x) + c * alpha * np.dot(grad_f(x), p):
        alpha *= c
    return alpha


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def rosenbrock_grad(x):
    grad = np.zeros_like(x)
    grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    grad[1] = 200 * (x[1] - x[0] ** 2)
    return grad


def second_function(x):
    return 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2

def sec_func_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = 300 * x[0] * x[1] ** 2 + 0.5 * x[0] + 2 * x[1] - 2
    grad[1] = 300 * x[0] ** 2 * x[1] + 2 * x[0] + 8 * x[1] - 8
    return grad


x0_rosenbrock = np.array([[1.2, 1.2], [-1.2, 1], [0.2, 0.8]], dtype=float)
x0_second_function = np.array([[-0.2, 1.2], [3.8, 0.1], [1.9, 0.6]], dtype=float)

rosenbrock_solutions = []
second_func_solutions = []

for values in x0_rosenbrock:
    solution = fletcher_reeves(rosenbrock, rosenbrock_grad, values)
    rosenbrock_solutions.append(solution)

for values in x0_second_function:
    solution = fletcher_reeves(second_function, sec_func_gradient, values)
    second_func_solutions.append(solution)

for idx, solution in enumerate(rosenbrock_solutions):
    print("Value pair:", idx + 1)
    print("x-values:", solution)
    print("Function value (Rosenbrock):", rosenbrock(solution))
    print()

for idx, solution in enumerate(second_func_solutions):
    print("Value pair:", idx + 1)
    print("x-values:", solution)
    print("Function value (Second function):", second_function(solution))
    print()

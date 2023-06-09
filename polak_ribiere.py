import numpy as np

def polak_ribiere(func, grad_func, x0, epsilon=1e-6, max_iter=50000):
    x = x0.astype(float)
    gradient = grad_func(x)
    p = -gradient
    step_size = 0.5
    c = 0.3
    k = 0

    while np.linalg.norm(gradient) > epsilon and k < max_iter:
        prev_gradient = gradient.copy()
        step_size = backtracking_line_search(func, grad_func, x, p, step_size, c)

        x = x + step_size * p
        gradient = grad_func(x)
        beta_pr = np.dot(gradient, gradient - prev_gradient) / np.dot(prev_gradient, prev_gradient)
        p = -gradient + beta_pr * p
        k += 1

    return x

def backtracking_line_search(f, grad_f, x, p, step_size, c):
    while f(x + step_size * p) > f(x) + c * step_size * np.dot(grad_f(x), p):
        step_size *= c
    return step_size

# Example usage
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    grad = np.zeros_like(x)
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    grad[1] = 200 * (x[1] - x[0]**2)
    return grad

def second_function(x):
    return x[0]**2 + x[1]**2

def sec_func_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = 2 * x[0]
    grad[1] = 2 * x[1]
    return grad

x0_rosenbrock = np.array([[1.2, 1.2], [-1.2, 1], [0.2, 0.8]], dtype=float)
x0_second_function = np.array([[-0.2, 1.2], [3.8, 0.1], [1.9, 0.6]], dtype=float)

rosenbrock_solutions = []
second_func_solutions = []

for values in x0_rosenbrock:
    solution = polak_ribiere(rosenbrock, rosenbrock_grad, values)
    rosenbrock_solutions.append(solution)

for values in x0_second_function:
    solution = polak_ribiere(second_function, sec_func_gradient, values)
    second_func_solutions.append(solution)

for idx, solution in enumerate(rosenbrock_solutions):
    print("value pair:", idx+1)
    print("x-values:", solution)
    print("function value (rosenbrock):", rosenbrock(solution))
    print()

for idx, solution in enumerate(second_func_solutions):
    print("value pair:", idx+1)
    print("x-values:", solution)
    print("function value (second function):", second_function(solution))
    print()

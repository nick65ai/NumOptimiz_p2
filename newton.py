import numpy as np

def rosenbrock_function(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_gradient(x):
    return np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]), 200 * (x[1] - x[0]**2)])

def rosenbrock_hessian(x):
    return np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])

def modified_newton_method(gradient, hessian, x0, epsilon=1e-6, max_iterations=100):
    x = x0
    iteration = 0

    while np.linalg.norm(gradient(x)) > epsilon and iteration < max_iterations:
        h_inv = np.linalg.inv(hessian(x))
        x = x - np.dot(h_inv, gradient(x))
        iteration += 1

    gradient_norm = np.linalg.norm(gradient(x))
    return x, gradient_norm, iteration

x0_list = [(1.2, 1.2), (-1.2, 1), (0.2, 0.8)]

print("Rosenbrock Function:")
for x0 in x0_list:
    x_opt, gradient_norm, iter = modified_newton_method(rosenbrock_gradient, rosenbrock_hessian, np.array(x0))
    distance = np.linalg.norm(x_opt - np.array([1, 1]))
    print(f"Starting point: {x0}")
    print(f"Optimal point: {x_opt}")
    print(f"Gradient norm: {gradient_norm}")
    print(f"Distance to solution: {distance}")
    print(f"Iterations: {iter}\n")

def function_2(x):
    return 150 * (x[0]*x[1])**2 + (0.5*x[0] + 2*x[1] - 2)**2

def gradient_2(x):
    return np.array([300*x[0]*(x[1])**2 + 0.5*x[0] + 2*x[1] -2,
                     300*x[1]*(x[0])**2 + 4*(0.5*x[0] + 2*x[1] - 2)])

def hessian_2(x):
    return np.array([[300 * (x[1])**2 + 0.5, 600 * x[0] * x[1] + 2],
                     [600 * x[1] * x[0] + 2, 300 * (x[0])**2 + 8]])


x0_list_2 = [(-0.2, 1.2), (3.8, 0.1), (1.9, 0.6)]

print("\nFunction 2:")
for x0 in x0_list_2:
    x_opt, gradient_norm, iter = modified_newton_method(gradient_2, hessian_2, np.array(x0))
    distance = [np.linalg.norm(x_opt - np.array([0, 1])), np.linalg.norm(x_opt - np.array([4, 0]))]
    print(f"Starting point: {x0}")
    print(f"Optimal point: {x_opt}")
    print(f"Gradient norm: {gradient_norm}")
    print(f"Distance to solution: {min(distance)}")
    print(f"Iterations: {iter}\n")

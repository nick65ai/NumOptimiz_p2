import matplotlib.pyplot as plt
import numpy as np
from autograd import grad, hessian, jacobian
from gradapprox import Approximation
from scipy.optimize import minimize

start = np.array([1.2, 1.2])


def f(x):
    return 100 * (x[1] - (x[0]) ** 2) ** 2 + (1 - x[0]) ** 2
    # return 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2


df = grad(f)

minimizer = minimize(f, start, method='BFGS').x

'''

   Line search

'''


def backtracking_line_search(f, df, x0, pk, alpha, rho):
    step = 1
    gradient_square_norm = np.linalg.norm(df(x0)) ** 2
    while f((x0 + (step * pk))) > f(x0) + (alpha * step * gradient_square_norm):
        step *= rho
    return step


# Function plot
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
z = np.zeros(([len(x1), len(x2)]))
for i in range(0, len(x1)):
    for j in range(0, len(x2)):
        z[j, i] = f([x1[i], x2[j]])

contours = plt.contour(x1, x2, z, 100, cmap=plt.cm.gnuplot)
plt.clabel(contours, inline=1, fontsize=10)
plt.xlabel("$x_1$ ->")
plt.ylabel("$x_2$ ->")

'''

   BFGS method

'''


def bfgs(xj, tolerance=1e-6, alpha=1e-4, rho=0.8, approx=False):
    x1 = [xj[0]]
    x2 = [xj[1]]
    bf = np.eye(len(xj))
    iters = 0

    while True:
        if approx is True:
            grad_f = lambda xj: Approximation(f).estimate_grad(xj[0], xj[1])
            gradient = grad_f(xj)
        else:
            gradient = df(xj)

        delta = -bf.dot(gradient)

        start_point = xj
        beta = backtracking_line_search(f, df, x0=start_point, pk=delta, alpha=alpha, rho=rho)
        if beta is not None:
            x = xj + beta * delta
        if np.linalg.norm(df(x)) < tolerance:
            x1 += [x[0], ]
            x2 += [x[1], ]
            plt.plot(x1, x2, "rx-", ms=5.5)
            plt.show()
            return x, f(x), iters, np.linalg.norm(df(xj)), abs(f(x) - minimizer)
        else:
            dj = x - xj
            gj = df(x) - gradient

            den = dj.dot(gj)
            num = bf.dot(gj)

            l = 1 + num.dot(gj) / den
            m = np.outer(dj, dj) / den
            n = np.outer(dj, num) / den
            o = np.outer(num, dj) / den

            delta = l * m - n - o
            bf += delta
            xj = x
            x1 += [xj[0], ]
            x2 += [xj[1], ]
        iters += 1


print(f'BFGS result:{bfgs(start)}\n')
print(f'BFGS result with approximation:{bfgs(start, approx=True)}\n')


# Function plot for SR1
contours = plt.contour(x1, x2, z, 100, cmap=plt.cm.gnuplot)
plt.clabel(contours, inline=1, fontsize=10)
plt.xlabel("$x_1$ ->")
plt.ylabel("$x_2$ ->")

'''

   SR1 method
   
'''


def sr1(xj, tolerance=1e-6, alpha=1e-4, rho=0.8, approx=False):
    x1 = [xj[0]]
    x2 = [xj[1]]
    bf = np.eye(len(xj))
    iters = 0
    while True:
        if approx is True:
            grad_f = lambda xj: Approximation(f).estimate_grad(xj[0], xj[1])
            gradient = grad_f(xj)
        else:
            gradient = df(xj)

        delta = -bf.dot(gradient)
        start_point = xj
        beta = backtracking_line_search(f, df, x0=start_point, pk=-gradient, alpha=alpha, rho=rho)
        if beta is not None:
            x = xj + beta * delta
        if np.linalg.norm(df(x)) < tolerance:
            x1 += [x[0], ]
            x2 += [x[1], ]
            plt.plot(x1, x2, "rx-", ms=5.5)
            plt.show()
            return x, f(x), iters, np.linalg.norm(df(xj)), abs(f(x) - minimizer)
        else:
            dj = x - xj
            gj = df(x) - gradient
            w = dj - bf.dot(gj)
            wT = np.transpose(w)
            sigma = 1 / (wT.dot(gj))
            W = np.outer(w, w)
            delta = sigma * W
            if abs(wT.dot(gj)) >= 1e-8 * np.linalg.norm(gj) * np.linalg.norm(w):
                bf += delta
            xj = x
            x1 += [xj[0], ]
            x2 += [xj[1], ]
        iters += 1


print(f'SR1 result:{sr1(start)}')
print(f'SR1 result with approximation:{sr1(start, approx=True)}')

'''

   SR1 with trust region
   - using dogleg method for subproblem solving

'''


def dogleg(hk, gk, bk, trust_radius):
    pB = -np.dot(hk, gk)
    norm_pB = np.sqrt(np.dot(pB, pB))

    if norm_pB <= trust_radius:
        return pB

    pU = - (np.dot(gk, gk) / np.dot(gk, np.dot(bk, gk))) * gk
    dot_pU = np.dot(pU, pU)
    norm_pU = np.sqrt(dot_pU)

    if norm_pU >= trust_radius:
        return trust_radius * pU / norm_pU

    pB_pU = pB - pU
    dot_pB_pU = np.dot(pB_pU, pB_pU)
    dot_pU_pB_pU = np.dot(pU, pB_pU)
    fact = dot_pU_pB_pU ** 2 - dot_pB_pU * (dot_pU - trust_radius ** 2)
    tau = (-dot_pU_pB_pU + np.sqrt(fact)) / dot_pB_pU

    return pU + tau * pB_pU


def sr1_trust_region(xj, hes, jac, trust_radius=1.0, eta=1e-4, tol=1e-6):
    k = 0
    iters = 0
    while True:
        bk = hes(xj)
        gk = jac(xj)
        hk = np.linalg.inv(bk)

        pk = dogleg(hk, gk, bk, trust_radius)
        ared = f(xj) - f(xj + pk)
        pred = -(np.dot(gk, pk) + 0.5 * np.dot(pk, np.dot(bk, pk)))
        rhok = ared / pred

        norm_pk = np.sqrt(np.dot(pk, pk))

        if rhok > eta:
            xj = xj + pk
        else:
            xj = xj

        if rhok > 0.75:
            if norm_pk <= 0.8 * trust_radius:
                trust_radius = trust_radius
            else:
                trust_radius = 2 * trust_radius
        elif 0.1 <= rhok <= 0.75:
            trust_radius = trust_radius
        else:
            trust_radius = 0.5 * trust_radius

        if np.linalg.norm(gk) < tol:
            break
        k = k + 1
        iters += 1

    return xj, f(xj), iters, np.linalg.norm(df(xj)), abs(f(xj) - minimizer)


print(f'SR1 with trust-region result:{sr1_trust_region(start, hessian(f), jacobian(f))}')

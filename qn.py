import matplotlib.pyplot as plt
import numpy as np
from autograd import grad


def f(x):
    # return 100 * (x[1] - (x[0]) ** 2) ** 2 + (1 - x[0]) ** 2
    return 150*(x[0]*x[1])**2 + (0.5*x[0] + 2*x[1] - 2)**2


df = grad(f)

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


def bfgs(xj, tolerance=1e-6, alpha=1e-4, rho=0.8):
    x1 = [xj[0]]
    x2 = [xj[1]]
    bf = np.eye(len(xj))

    while True:
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
            return x, f(x)
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


print(f'BFGS result:{bfgs(np.array([1.2, 1.2]))}\n')

# Function plot for SR1
contours = plt.contour(x1, x2, z, 100, cmap=plt.cm.gnuplot)
plt.clabel(contours, inline=1, fontsize=10)
plt.xlabel("$x_1$ ->")
plt.ylabel("$x_2$ ->")

'''

   SR1 method
   
'''


def sr1(xj, tolerance=1e-6, alpha=1e-4, rho=0.8):
    x1 = [xj[0]]
    x2 = [xj[1]]
    bf = np.eye(len(xj))

    while True:
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
            return x, f(x)
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


print(f'SR1 result:{sr1(np.array([-1.2, 1]))}')

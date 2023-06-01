from scipy.linalg import hilbert
import numpy as np


def conjugate_grad(dim):
    A = hilbert(dim)
    b = np.ones(dim)
    x = np.zeros(dim)

    # initializing x0, r0, p0

    r = A @ x - b
    p = -r
    k = 0

    # algorithm
    
    while True:
        a = (r.T @ r) / (p.T @ A @ p)
        x1 = x + a * p
        r1 = r + a * A @ p
        b1 = (r1.T @ r1) / (r.T @ r)
        p1 = -r1 + b1 * p
        k += 1

        # if residual is below the given treshold
        if np.linalg.norm(r) <= 10 ** -6:
            break
        x, r, p = x1, r1, p1

    return x1, ke

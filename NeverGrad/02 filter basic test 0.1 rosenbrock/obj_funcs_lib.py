import numpy as np

def sphere(X):
    """
    Sphere Function
    f(x) = sum(x_i^2)
    Global minimum at 0.
    """
    X = np.asarray(X)
    return np.sum(X ** 2, axis=-1)


def rosenbrock(X, a=1, b=100):
    """
    Rosenbrock Function (Banana Function)
    f(x) = sum_{i=1}^{n-1} [b*(x_{i+1} - x_i^2)^2 + (a - x_i)^2]
    Global minimum at [a, ..., a].
    """
    X = np.asarray(X)
    X = np.atleast_2d(X)
    return np.sum(b * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (a - X[..., :-1]) ** 2, axis=-1)


def rastrigin(X):
    """
    Rastrigin Function
    f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global minimum at 0.
    """
    X = np.asarray(X)
    n = X.shape[-1] if X.ndim > 1 else X.size
    return 10 * n + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=-1)


def ackley(X, a=20, b=0.2, c=2 * np.pi):
    """
    Ackley Function
    f(x) = -a*exp(-b*sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c*x_i))) + a + exp(1)
    Global minimum at 0.
    """
    X = np.asarray(X)
    X = np.atleast_2d(X)
    n = X.shape[-1]
    sum_sq = np.sum(X ** 2, axis=-1)
    sum_cos = np.sum(np.cos(c * X), axis=-1)
    return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.exp(1)


def griewank(X):
    """
    Griewank Function
    f(x) = 1 + 1/4000*sum(x_i^2) - prod(cos(x_i/sqrt(i+1)))
    Global minimum at 0.
    """
    X = np.asarray(X)
    X = np.atleast_2d(X)
    sum_sq = np.sum(X ** 2, axis=-1) / 4000.0
    i = np.arange(1, X.shape[-1] + 1)
    prod_cos = np.prod(np.cos(X / np.sqrt(i)), axis=-1)
    return sum_sq - prod_cos + 1


# Dictionary mapping function names to functions for convenience
OBJ_FUNCS = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "griewank": griewank,
}

__all__ = [
    "sphere", "rosenbrock", "rastrigin", "ackley", "griewank",
    "OBJ_FUNCS"
]


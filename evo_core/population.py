"""Инициализация популяции и учёт вызовов фитнес-функции."""
import numpy as np


def init_uniform(rng, n, lower, upper):
    """n точек, равномерно распределённых в боксе [lower, upper]."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return lower + rng.random((n, lower.size)) * (upper - lower)


class Evaluator:
    """Обёртка над ФФ: считает вызовы и не даёт превысить бюджет."""

    def __init__(self, func, budget):
        self.func = func
        self.budget = int(budget)
        self.calls = 0

    @property
    def remaining(self):
        return self.budget - self.calls

    def __call__(self, X):
        X = np.atleast_2d(X)
        if self.calls + len(X) > self.budget:
            raise RuntimeError("бюджет вычислений ФФ исчерпан")
        self.calls += len(X)
        return self.func(X)


def init_binary(rng, n, length, p_one):
    """n бинарных строк; каждый бит равен 1 с вероятностью p_one."""
    return (rng.random((n, length)) < p_one).astype(np.int8)

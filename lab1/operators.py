"""ЛР1: целевая функция и операторы вещественного ГА."""
import numpy as np


def griewank(X):
    """Функция Гриванка, векторизована по строкам X (n, d). Минимум f(0) = 0."""
    X = np.atleast_2d(X)
    i = np.arange(1, X.shape[1] + 1)
    return 1.0 + np.sum(X ** 2, axis=1) / 4000.0 - np.prod(np.cos(X / np.sqrt(i)), axis=1)


PROBLEMS = {"griewank": griewank}


def blx_alpha(rng, P1, P2, alpha):
    """BLX-α: ген потомка равномерен на [min − α·I, max + α·I], I = |p1 − p2|."""
    lo = np.minimum(P1, P2)
    hi = np.maximum(P1, P2)
    span = hi - lo
    return rng.uniform(lo - alpha * span, hi + alpha * span)


def uniform_crossover(rng, P1, P2):
    """Равномерный (дискретный) кроссовер: каждый ген от случайного родителя."""
    mask = rng.random(P1.shape) < 0.5
    return np.where(mask, P1, P2), np.where(mask, P2, P1)


def gaussian_mutation(rng, X, sigma, p_mut):
    """Каждый ген с вероятностью p_mut получает сдвиг N(0, sigma²); sigma — скаляр или (n, 1)."""
    mask = rng.random(X.shape) < p_mut
    return X + mask * rng.normal(size=X.shape) * sigma


def self_adapt_sigma(rng, sigma, tau, s_min, s_max):
    """Самоадаптация шага: σ' = σ·exp(τ·N(0,1)) — логнормальное правило Швефеля."""
    return np.clip(sigma * np.exp(tau * rng.normal(size=sigma.shape)), s_min, s_max)

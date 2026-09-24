"""ЛР3: вещественные операторы для NSGA-II — SBX и полиномиальная мутация (Deb)."""
import numpy as np


def sbx(rng, P1, P2, eta, lower, upper, p_gene=0.5):
    """Имитация двоичного кроссовера: потомки симметричны относительно среднего родителей,
    разброс задаёт η (больше η — ближе к родителям). Каждый ген скрещивается с вероятностью p_gene."""
    u = rng.random(P1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    mix = rng.random(P1.shape) < p_gene
    beta = np.where(mix, beta, 1.0)
    C1 = 0.5 * ((1 + beta) * P1 + (1 - beta) * P2)
    C2 = 0.5 * ((1 - beta) * P1 + (1 + beta) * P2)
    return np.clip(C1, lower, upper), np.clip(C2, lower, upper)


def polynomial_mutation(rng, X, eta, lower, upper, p_gene):
    """Полиномиальная мутация с учётом границ: результат всегда в [lower, upper]."""
    X = X.copy()
    mask = rng.random(X.shape) < p_gene
    span = upper - lower
    d1, d2 = (X - lower) / span, (upper - X) / span
    u = rng.random(X.shape)
    pw = 1 / (eta + 1)
    left = (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** pw - 1
    right = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** pw
    delta = np.where(u < 0.5, left, right)
    X[mask] = (X + delta * span)[mask]
    return np.clip(X, lower, upper)

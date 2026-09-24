"""Селекция родителей. Везде минимизация: меньше фитнес — лучше."""
import numpy as np


def tournament(rng, fitness, n_select, k):
    """Турнир размера k с возвращением; k управляет селективным давлением."""
    idx = rng.integers(0, len(fitness), size=(n_select, k))
    winners = np.argmin(fitness[idx], axis=1)
    return idx[np.arange(n_select), winners]


def linear_rank_probs(fitness, s=1.5):
    """Линейное ранжирование Бейкера, s ∈ [1, 2] — ожидаемое число потомков лучшей особи."""
    n = len(fitness)
    ranks = np.empty(n, dtype=int)
    ranks[np.argsort(fitness, kind="stable")] = np.arange(n)  # 0 — лучшая
    return (2 - s) / n + 2 * (s - 1) * (n - 1 - ranks) / (n * (n - 1))


def rank_selection(rng, fitness, n_select, s=1.5):
    return rng.choice(len(fitness), size=n_select, p=linear_rank_probs(fitness, s))

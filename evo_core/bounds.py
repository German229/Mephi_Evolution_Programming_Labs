"""Обработка выхода за границы бокса."""
import numpy as np


def reflect(X, lower, upper):
    """Зеркальное отражение от границ; корректно и при выходе дальше ширины области."""
    width = upper - lower
    y = np.mod(X - lower, 2 * width)
    y = np.where(y > width, 2 * width - y, y)
    return lower + y


def clip(X, lower, upper):
    return np.clip(X, lower, upper)

"""Элитизм: e лучших особей переходят в следующее поколение без изменений."""
import numpy as np


def elite_indices(fitness, e):
    return np.argsort(fitness, kind="stable")[:e]

"""Агрегаты серий, запись CSV и непараметрический тест Манна—Уитни."""
import csv
import math
from pathlib import Path

import numpy as np


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def fmt(x):
    """Компактная запись чисел в CSV."""
    return f"{x:.6g}" if isinstance(x, (float, np.floating)) else x


def describe(values):
    v = np.asarray(values, dtype=float)
    return {
        "best": v.min(), "mean": v.mean(), "median": np.median(v),
        "std": v.std(ddof=1) if v.size > 1 else 0.0, "worst": v.max(),
    }


def mann_whitney(a, b):
    """Двусторонний U-тест, нормальная аппроксимация с поправкой на связки.

    Возвращает (U, z, p). Отрицательный z — значения a систематически меньше b.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = a.size, b.size
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty_like(allv)
    sorted_v = allv[order]
    i = 0
    while i < allv.size:  # средние ранги для связок
        j = i
        while j + 1 < allv.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    n = n1 + n2
    _, counts = np.unique(allv, return_counts=True)
    tie = (counts ** 3 - counts).sum() / (n * (n - 1))
    sigma = math.sqrt(n1 * n2 / 12 * ((n + 1) - tie))
    if sigma == 0:
        return u1, 0.0, 1.0
    z = (u1 - n1 * n2 / 2) / sigma
    return u1, z, math.erfc(abs(z) / math.sqrt(2))


def vargha_delaney_a12(a, b):
    """Размер эффекта Â12 = P(A > B) + 0.5·P(A = B); 0.5 — нет эффекта, 0.71+ — «большой» (Vargha, Delaney 2000)."""
    a = np.asarray(a, float)[:, None]
    b = np.asarray(b, float)[None, :]
    return float((a > b).mean() + 0.5 * (a == b).mean())


def holm(pvalues):
    """Поправка Холма—Бонферрони: скорректированные p-значения в исходном порядке."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj

"""Многокритериальные примитивы (минимизация всех критериев): доминирование Парето,
быстрая недоминируемая сортировка, расстояние скученности, гиперобъём."""
import numpy as np


def dominance_matrix(F):
    """D[i, j] = True, если решение i доминирует j (∀k F_ik ≤ F_jk и ∃k F_ik < F_jk)."""
    le = (F[:, None, :] <= F[None, :, :]).all(-1)
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    return le & lt


def non_dominated_sort(F):
    """Быстрая недоминируемая сортировка (Deb et al., 2002).

    Возвращает (ranks, fronts): ранг каждой точки (0 — фронт Парето) и список индексов по фронтам.
    """
    n = len(F)
    D = dominance_matrix(F)
    dominated_by = D.sum(0)            # сколько решений доминирует j
    ranks = np.full(n, -1)
    fronts, current, r = [], np.flatnonzero(dominated_by == 0), 0
    while current.size:
        ranks[current] = r
        fronts.append(current)
        dominated_by = dominated_by - D[current].sum(0)
        current = np.flatnonzero((dominated_by == 0) & (ranks == -1))
        r += 1
    return ranks, fronts


def nondominated(F):
    """Индексы недоминируемых точек."""
    return np.flatnonzero(~dominance_matrix(F).any(0))


def crowding_distance(F):
    """Расстояние скученности внутри одного фронта; крайние точки получают ∞."""
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        order = np.argsort(F[:, k], kind="stable")
        span = F[order[-1], k] - F[order[0], k]
        d[order[0]] = d[order[-1]] = np.inf
        if span > 0:
            d[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
    return d


def crowded_tournament(rng, ranks, crowd, n_select):
    """Бинарный турнир по отношению «ранг меньше, при равенстве — скученность больше»."""
    a = rng.integers(0, len(ranks), n_select)
    b = rng.integers(0, len(ranks), n_select)
    a_wins = (ranks[a] < ranks[b]) | ((ranks[a] == ranks[b]) & (crowd[a] > crowd[b]))
    return np.where(a_wins, a, b)


def survival(F, n):
    """Отбор n лучших из объединения P ∪ Q: целые фронты, последний — по убыванию скученности."""
    ranks, fronts = non_dominated_sort(F)
    crowd = np.zeros(len(F))
    keep = []
    for fr in fronts:
        cd = crowding_distance(F[fr])
        crowd[fr] = cd
        if len(keep) + len(fr) <= n:
            keep.extend(fr)
        else:
            keep.extend(fr[np.argsort(-cd, kind="stable")[: n - len(keep)]])
            break
    keep = np.array(keep)
    return keep, ranks[keep], crowd[keep]


def _hv2d(P, ref):
    """Площадь, доминируемая точками P (n, 2) и ограниченная ref: сумма полос при обходе по x."""
    P = P[np.all(P < ref, axis=1)]
    area, best_y = 0.0, ref[1]
    for x, y in P[np.lexsort((P[:, 1], P[:, 0]))]:
        if y < best_y:
            area += (ref[0] - x) * (best_y - y)
            best_y = y
    return area


def hypervolume(F, ref):
    """Точный гиперобъём для 2 или 3 критериев (3D — послойно по третьему критерию)."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    if F.shape[1] == 2:
        return _hv2d(F, ref)
    if F.shape[1] != 3:
        raise ValueError("поддерживаются 2 или 3 критерия")
    F = F[np.argsort(F[:, 2], kind="stable")]
    zs = np.append(F[:, 2], ref[2])
    vol = 0.0
    for k in range(len(F)):
        dz = zs[k + 1] - zs[k]
        if dz > 0:
            vol += _hv2d(F[:k + 1, :2], ref[:2]) * dz
    return vol

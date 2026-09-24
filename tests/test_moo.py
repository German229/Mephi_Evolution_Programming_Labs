import numpy as np

from evo_core.moo import (crowding_distance, dominance_matrix, hypervolume, non_dominated_sort,
                          nondominated)


def brute_ranks(F):
    ranks, left, r = np.full(len(F), -1), set(range(len(F))), 0
    while left:
        front = [i for i in left if not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i]) for j in left)]
        ranks[front] = r
        left -= set(front)
        r += 1
    return ranks


def test_non_dominated_sort_matches_brute_force():
    rng = np.random.default_rng(0)
    for m in (2, 3):
        F = rng.integers(0, 6, size=(60, m)).astype(float)  # много равенств и дубликатов
        ranks, fronts = non_dominated_sort(F)
        assert np.array_equal(ranks, brute_ranks(F))
        assert sorted(np.concatenate(fronts).tolist()) == list(range(60))


def test_dominance_basic():
    D = dominance_matrix(np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [0.0, 3.0]]))
    assert D[0, 1] and not D[0, 2] and not D[3, 0] and not D[0, 3]
    assert set(nondominated(np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 3.0]]))) == {0, 2}


def test_crowding_extremes_infinite():
    d = crowding_distance(np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]))
    assert np.isinf(d[0]) and np.isinf(d[3]) and np.allclose(d[1:3], 4 / 3)


def test_hypervolume_known_values():
    assert np.isclose(hypervolume([[0.5, 0.5]], [1, 1]), 0.25)
    assert np.isclose(hypervolume([[0.2, 0.8], [0.8, 0.2]], [1, 1]), 0.28)
    assert np.isclose(hypervolume([[0, 0, 0.5], [0.5, 0.5, 0]], [1, 1, 1]), 0.625)
    assert hypervolume([[2.0, 0.0, 0.0]], [1, 1, 1]) == 0.0


def test_hypervolume_monte_carlo_3d():
    rng = np.random.default_rng(1)
    P = rng.random((15, 3))
    S = rng.random((200000, 3))
    mc = np.mean((S[:, None, :] >= P[None, :, :]).all(-1).any(1))
    assert abs(hypervolume(P, [1, 1, 1]) - mc) < 5e-3

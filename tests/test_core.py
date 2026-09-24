import numpy as np
import pytest

from evo_core.bounds import reflect
from evo_core.elitism import elite_indices
from evo_core.population import Evaluator, init_uniform
from evo_core.selection import linear_rank_probs, rank_selection, tournament
from evo_core.stats import describe, mann_whitney


def test_init_inside_bounds():
    rng = np.random.default_rng(0)
    X = init_uniform(rng, 500, [-600] * 10, [600] * 10)
    assert X.shape == (500, 10) and X.min() >= -600 and X.max() <= 600


def test_evaluator_counts_and_enforces_budget():
    ev = Evaluator(lambda X: X.sum(1), budget=10)
    ev(np.zeros((7, 2)))
    assert ev.calls == 7 and ev.remaining == 3
    with pytest.raises(RuntimeError):
        ev(np.zeros((4, 2)))


@pytest.mark.parametrize("scale", [1, 5, 1000])
def test_reflect_keeps_points_in_box(scale):
    rng = np.random.default_rng(1)
    lo, hi = np.full(3, -2.0), np.full(3, 3.0)
    Y = reflect(rng.normal(size=(1000, 3)) * scale, lo, hi)
    assert Y.min() >= -2.0 and Y.max() <= 3.0


def test_reflect_identity_inside_and_mirror():
    lo, hi = np.array([0.0]), np.array([10.0])
    assert np.allclose(reflect(np.array([[3.0]]), lo, hi), 3.0)
    assert np.allclose(reflect(np.array([[-2.0]]), lo, hi), 2.0)
    assert np.allclose(reflect(np.array([[13.0]]), lo, hi), 7.0)


def test_tournament_indices_valid_and_pressure():
    rng = np.random.default_rng(2)
    F = np.arange(50, dtype=float)
    idx = tournament(rng, F, 10000, k=3)
    assert idx.min() >= 0 and idx.max() < 50
    assert F[idx].mean() < F.mean()  # отбор смещён к лучшим


def test_rank_probs_sum_and_order():
    p = linear_rank_probs(np.array([5.0, 1.0, 3.0]), s=1.5)
    assert np.isclose(p.sum(), 1) and p[1] > p[2] > p[0]
    idx = rank_selection(np.random.default_rng(3), np.array([5.0, 1.0, 3.0]), 100)
    assert set(idx) <= {0, 1, 2}


def test_elite_indices():
    assert list(elite_indices(np.array([3.0, 1.0, 2.0, 0.5]), 2)) == [3, 1]


def test_describe_and_mann_whitney():
    d = describe([1, 2, 3])
    assert d["best"] == 1 and d["worst"] == 3 and d["median"] == 2
    _, z, p = mann_whitney(np.arange(20), np.arange(20) + 100)
    assert z < 0 and p < 1e-6
    _, _, p_same = mann_whitney(np.ones(10), np.ones(10))
    assert p_same == 1.0


def test_a12_and_holm():
    from evo_core.stats import holm, vargha_delaney_a12
    assert vargha_delaney_a12([2, 3], [0, 1]) == 1.0 and vargha_delaney_a12([1, 1], [1, 1]) == 0.5
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

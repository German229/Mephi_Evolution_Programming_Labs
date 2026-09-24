import numpy as np

from evo_core.cli import load_config
from lab3.model import load_model
from evo_core.moo import survival
from lab3.nsga2 import run_nsga2, run_weighted_sum, weight_vectors
from lab3.operators import polynomial_mutation, sbx

CFG = "lab3/configs/nsga2.toml"
M = load_model()


def test_water_mass_balance():
    rng = np.random.default_rng(0)
    I = rng.uniform(0, M.i_max, (50, M.days))
    _, tr = M.simulate(I, trace=True)
    inflow = M.rain.sum(1)[None, :] + M.efficiency * I.sum(1)[:, None]      # (pop, K)
    out = tr["et"].sum(-1) + tr["runoff"].sum(-1) + tr["drainage"].sum(-1)
    assert tr["theta"].shape == (50, M.scenarios, M.days)
    assert np.allclose(M.theta0 + inflow - out, tr["theta"][..., -1])
    assert (tr["theta"] >= 0).all() and (tr["theta"] <= M.theta_sat).all()


def test_objectives_monotone_extremes():
    F = M.simulate(np.vstack([np.zeros(M.days), np.full(M.days, M.i_max)]))
    assert F[0, 0] == 0 and F[0, 1] > F[1, 1] and F[1, 2] > F[0, 2]


def test_operators_respect_bounds():
    rng = np.random.default_rng(1)
    lo, hi = np.zeros(M.days), np.full(M.days, M.i_max)
    P1, P2 = rng.uniform(0, M.i_max, (200, M.days)), rng.uniform(0, M.i_max, (200, M.days))
    for C in sbx(rng, P1, P2, 2.0, lo, hi):
        assert C.min() >= 0 and C.max() <= M.i_max
    X = polynomial_mutation(rng, P1, 1.0, lo, hi, 1.0)
    assert X.min() >= 0 and X.max() <= M.i_max and not np.array_equal(X, P1)


def test_sbx_preserves_parent_mean():
    rng = np.random.default_rng(2)
    P1, P2 = np.full((10, 5), 5.0), np.full((10, 5), 15.0)
    C1, C2 = sbx(rng, P1, P2, 10.0, np.zeros(5), np.full(5, 100.0), p_gene=1.0)
    assert np.allclose((C1 + C2) / 2, 10.0)


def test_survival_keeps_first_front_and_extremes():
    F = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0], [3.0, 3.0], [2.5, 2.5]])
    keep, ranks, _ = survival(F, 3)
    assert set(keep) <= {0, 1, 2, 3} and {0, 3} <= set(keep) and (ranks == 0).all()


def test_weight_vectors_simplex():
    W = weight_vectors(3)
    assert len(W) == 10 and np.allclose(W.sum(1), 1) and (W >= 0).all()


def test_nsga2_reproducible_budget_and_nondominated():
    cfg = load_config(CFG)
    cfg["moea"]["budget"] = 1000
    r1, r2 = run_nsga2(cfg, 3), run_nsga2(cfg, 3)
    assert r1["evals"] == 1000 and np.array_equal(r1["F"], r2["F"])
    D = (r1["F"][:, None, :] <= r1["F"][None]).all(-1) & (r1["F"][:, None, :] < r1["F"][None]).any(-1)
    assert not D.any()
    assert run_weighted_sum(cfg, 3)["evals"] == 1000

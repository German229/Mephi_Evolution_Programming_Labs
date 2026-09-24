import numpy as np

from evo_core.cli import load_config
from lab1.ga import run_ga, run_random_search
from lab1.operators import (blx_alpha, gaussian_mutation, griewank, self_adapt_sigma,
                            uniform_crossover)

CFG = "lab1/configs/ga_base.toml"


def test_griewank_known_values():
    assert np.isclose(griewank(np.zeros((1, 10)))[0], 0.0, atol=1e-15)
    x = np.array([[1.0, 2.0]])  # ручной расчёт
    ref = 1 + 5 / 4000 - np.cos(1.0) * np.cos(2 / np.sqrt(2))
    assert np.isclose(griewank(x)[0], ref)
    assert np.all(griewank(np.random.default_rng(0).uniform(-600, 600, (100, 10))) >= 0)


def test_blx_range():
    rng = np.random.default_rng(0)
    P1, P2 = np.zeros((1000, 4)), np.ones((1000, 4))
    C = blx_alpha(rng, P1, P2, 0.5)
    assert C.min() >= -0.5 and C.max() <= 1.5


def test_uniform_crossover_takes_genes_from_parents():
    rng = np.random.default_rng(1)
    P1, P2 = np.zeros((50, 6)), np.ones((50, 6))
    C1, C2 = uniform_crossover(rng, P1, P2)
    assert np.all((C1 == 0) | (C1 == 1)) and np.all(C1 + C2 == 1)


def test_mutation_probability_zero_is_identity():
    X = np.random.default_rng(2).normal(size=(10, 5))
    assert np.array_equal(gaussian_mutation(np.random.default_rng(3), X, 1.0, 0.0), X)


def test_self_adapt_sigma_bounds():
    s = self_adapt_sigma(np.random.default_rng(4), np.full(10000, 0.01), 3.0, 1e-6, 0.5)
    assert s.min() >= 1e-6 and s.max() <= 0.5


def test_ga_respects_bounds_budget_and_is_reproducible():
    cfg = load_config(CFG)
    cfg["ga"]["budget"] = 3000
    cfg["ga"]["sigma_frac"] = 0.5  # крупные шаги часто выводят за границы до ремонта
    r1, r2 = run_ga(cfg, 7), run_ga(cfg, 7)
    assert r1["evals"] == 3000 and r1["feasible"]
    assert np.all(np.abs(r1["best_x"]) <= 600)
    assert r1["best_f"] == r2["best_f"] and np.array_equal(r1["best_x"], r2["best_x"])
    assert run_ga(cfg, 8)["best_f"] != r1["best_f"]


def test_best_so_far_monotone():
    cfg = load_config(CFG)
    cfg["ga"]["budget"] = 5000
    log = np.array(run_ga(cfg, 0)["log"])
    assert np.all(np.diff(log[:, 2]) <= 0)


def test_random_search_uses_exact_budget():
    cfg = load_config("lab1/configs/random_search.toml")
    cfg["ga"]["budget"] = 1234
    assert run_random_search(cfg, 0)["evals"] == 1234

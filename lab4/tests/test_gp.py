import numpy as np
import pytest

from evo_core.cli import load_config
from lab4 import data
from lab4.gp import BAD, Fitness, run_gp
from lab4.tree import (ARITY, Language, depth, point_mutation, predict, subtree_crossover, subtree_end,
                       subtree_mutation, to_infix)

LANG = Language(list(ARITY), 2)


def valid(tree):
    """Синтаксическая корректность префиксной записи: дерево ровно покрывает список."""
    return len(tree) > 0 and subtree_end(tree, 0) == len(tree)


def test_random_trees_valid_and_depth_bounded():
    rng = np.random.default_rng(0)
    for t in LANG.ramped_half_and_half(rng, 200, 2, 6):
        assert valid(t) and depth(t) <= 6


@pytest.mark.parametrize("seed", range(5))
def test_operators_keep_syntax_and_limits(seed):
    rng = np.random.default_rng(seed)
    pop = LANG.ramped_half_and_half(rng, 50, 2, 6)
    for k in range(0, 50, 2):
        for c in subtree_crossover(rng, pop[k], pop[k + 1], 8, 40):
            assert valid(c) and depth(c) <= max(8, depth(pop[k]), depth(pop[k + 1]))
        m = subtree_mutation(rng, pop[k], LANG, 17, 500)
        p = point_mutation(rng, pop[k], LANG)
        assert valid(m) and valid(p) and len(p) == len(pop[k])


def test_crossover_rejects_oversize():
    rng = np.random.default_rng(1)
    big = LANG.random_tree(rng, 6, "full")
    c1, _ = subtree_crossover(rng, big, big, 17, len(big))
    assert len(c1) <= len(big)


def test_protected_ops_finite_and_known_values():
    X = np.array([[0.0, 0.0], [2.0, -1.0]])
    t = [("f", "add"), ("f", "div"), ("x", 0), ("x", 0), ("f", "log"), ("x", 1)]  # x0/x0 + log(x1)
    assert np.allclose(predict(t, X), [1.0, 1.0])
    assert to_infix(t) == "(x0 / x0) + log(x1)"


def test_linear_scaling_recovers_affine_target():
    X, y, m = data.load("nguyen7")
    f = Fitness(X, 3.0 * X[:, 0] - 2.0, m, 10)   # цель — аффинная функция x0
    tr, va, te, ex, a, b = f.errors([("x", 0)])
    assert tr < 1e-9 and np.isclose(a, -2.0) and np.isclose(b, 3.0)
    assert f.errors([("f", "exp"), ("c", 1000.0)])[0] <= BAD


@pytest.mark.parametrize("mode", ["depth", "parsimony", "tarpeian", "nsga2"])
def test_run_gp_budget_and_reproducible(mode):
    cfg = load_config("lab4/configs/gp_depth.toml")
    cfg["bloat_control"] = mode
    cfg["gp"].update(budget=600, pop_size=40, parsimony_c=1e-3, tarpeian_p=0.6)
    r1, r2 = run_gp(cfg, 1, "nguyen7"), run_gp(cfg, 1, "nguyen7")
    assert r1["evals"] == 600 and r1["expr"] == r2["expr"] and r1["test"] == r2["test"]
    assert valid(r1["tree"]) and r1["depth"] <= 17 and r1["size"] <= 500

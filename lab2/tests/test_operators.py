import itertools

import numpy as np
import pytest

from evo_core.cli import load_config
from lab2.exact import solve_exact
from lab2.ga import run_ga, run_random_feasible
from lab2.operators import (Repairer, bitflip_mutation, penalized_fitness, swap_mutation,
                            two_point_crossover, uniform_crossover)
from lab2.problem import Instance, is_feasible, load_instance, violations

INST = load_instance()


def random_strings(n_rows, p, seed=0):
    return (np.random.default_rng(seed).random((n_rows, INST.n)) < p).astype(np.int8)


@pytest.mark.parametrize("fill", [True, False])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 1.0])
def test_repair_always_feasible(fill, p):
    X = Repairer(INST, fill=fill)(random_strings(300, p, seed=int(p * 10)))
    assert is_feasible(INST, X).all()


def test_repair_keeps_feasible_unchanged_without_fill():
    rep = Repairer(INST, fill=False)
    x = Repairer(INST, fill=True).repair_one(np.zeros(INST.n, dtype=np.int8))
    assert np.array_equal(rep.repair_one(x), x)


def test_repair_resolves_each_constraint_type():
    rep = Repairer(INST, fill=False)
    x = np.zeros(INST.n, dtype=np.int8)
    x[[1, 2, 39]] = 1                      # конфликт 1–2
    assert violations(INST, x)["conflicts"][0] == 1 and is_feasible(INST, rep.repair_one(x))[0]
    x = np.zeros(INST.n, dtype=np.int8)
    x[[24, 22]] = 1                        # 22 требует 23 -> удаляется 22, затем 24 (транзитивно)
    y = rep.repair_one(x)
    assert y[22] == 0 and y[24] == 0


def test_crossovers_take_genes_from_parents():
    rng = np.random.default_rng(1)
    P1, P2 = random_strings(50, 0.5, 1), random_strings(50, 0.5, 2)
    for cx in (uniform_crossover, two_point_crossover):
        C1, C2 = cx(rng, P1, P2)
        assert np.all((C1 == P1) | (C1 == P2)) and np.array_equal(C1 + C2, P1 + P2)


def test_mutations_keep_binary_and_swap_keeps_count():
    rng = np.random.default_rng(2)
    X = random_strings(100, 0.3, 3)
    B = bitflip_mutation(rng, X, 0.05)
    S = swap_mutation(rng, X, 1.0)
    assert set(np.unique(B)) <= {0, 1} and set(np.unique(S)) <= {0, 1}
    assert np.array_equal(S.sum(1), X.sum(1)) and not np.array_equal(S, X)


def test_penalty_zero_for_feasible_positive_for_infeasible():
    x = Repairer(INST)(random_strings(5, 0.5))
    f, ok = penalized_fitness(INST, x, 300.0)
    assert ok.all() and np.allclose(f, -(x @ INST.value))
    f2, ok2 = penalized_fitness(INST, np.ones((1, INST.n), dtype=np.int8), 300.0)
    assert not ok2[0] and f2[0] > -(INST.value.sum())


def _sub_instance(k):
    keep = set(range(k))
    req = np.array([r for r in INST.req if set(r) <= keep]).reshape(-1, 2)
    conf = np.array([c for c in INST.conf if set(c) <= keep]).reshape(-1, 2)
    return Instance(INST.names[:k], INST.categories[:k], INST.cost[:k], INST.power[:k], INST.value[:k],
                    INST.cost[:k].sum() * 0.35, INST.power[:k].sum() * 0.35, req, conf, {})


def test_exact_matches_brute_force():
    inst = _sub_instance(16)
    X = np.array(list(itertools.product([0, 1], repeat=inst.n)), dtype=np.int8)
    ok = is_feasible(inst, X)
    brute = (X[ok] @ inst.value).max()
    assert np.isclose(solve_exact(inst)["value"], brute)


def test_ga_reproducible_feasible_and_budget():
    cfg = load_config("lab2/configs/ga_repair.toml")
    cfg["ga"]["budget"] = 1500
    r1, r2 = run_ga(cfg, 5, INST), run_ga(cfg, 5, INST)
    assert r1["evals"] == 1500 and r1["feasible"] and is_feasible(INST, r1["best_x"])[0]
    assert r1["best_value"] == r2["best_value"]
    cfg["ga"].update(constraints="penalty", penalty_lambda=300.0)
    rp = run_ga(cfg, 5, INST)
    assert rp["evals"] == 1500 and (not rp["feasible"] or is_feasible(INST, rp["best_x"])[0])


def test_random_feasible_only_feasible():
    cfg = load_config("lab2/configs/random_feasible.toml")
    cfg["ga"]["budget"] = 300
    r = run_random_feasible(cfg, 0, INST)
    assert r["evals"] == 300 and is_feasible(INST, r["best_x"])[0]

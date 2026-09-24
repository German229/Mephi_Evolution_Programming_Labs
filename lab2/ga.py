"""ЛР2: ГА с бинарным представлением (ремонт или штраф) и базовые методы сравнения."""
import time

import numpy as np

from evo_core.elitism import elite_indices
from evo_core.population import Evaluator, init_binary
from evo_core.selection import tournament
from lab2.operators import (Repairer, bitflip_mutation, penalized_fitness, swap_mutation,
                            two_point_crossover, uniform_crossover)
from lab2.problem import load_instance

LOG_COLUMNS = ["gen", "evals", "best_so_far", "pop_best", "pop_mean", "feasible_frac"]
CROSSOVERS = {"uniform": uniform_crossover, "two_point": two_point_crossover}


class Fitness:
    """Минимизируемый фитнес и признак допустимости; вызов считается Evaluator-ом."""

    def __init__(self, inst, g):
        self.inst, self.mode, self.lam = inst, g["constraints"], g.get("penalty_lambda", 0.0)
        self.repair = Repairer(inst, fill=g.get("repair_fill", True)) if self.mode == "repair" else None
        self.feasible = None

    def __call__(self, X):
        if self.mode == "repair":
            self.feasible = np.ones(len(X), dtype=bool)
            return -(X @ self.inst.value).astype(float)
        f, self.feasible = penalized_fitness(self.inst, X, self.lam)
        return f


def _mutate(rng, X, g):
    if g["mutation"] == "bitflip":
        return bitflip_mutation(rng, X, g["p_mutation"])
    if g["mutation"] == "swap":
        return swap_mutation(rng, X, g["p_mutation"])
    raise ValueError(g["mutation"])


def run_ga(cfg, seed, inst=None):
    g = cfg["ga"]
    inst = inst or load_instance(cfg["instance"])
    rng = np.random.default_rng(seed)
    fit = Fitness(inst, g)
    ev = Evaluator(fit, g["budget"])
    n, e = g["pop_size"], g["elite"]
    t0 = time.perf_counter()

    def evaluate(X):
        if fit.repair is not None:
            X = fit.repair(X)  # ламарковский ремонт до оценки
        return X, ev(X), fit.feasible.copy()

    X, F, feas = evaluate(init_binary(rng, n, inst.n, g["p_init"]))
    best_v, best_x = -np.inf, None
    log, gen = [], 0

    def update_best():
        nonlocal best_v, best_x
        if feas.any():
            vals = X @ inst.value
            i = np.flatnonzero(feas)[np.argmax(vals[feas])]
            if vals[i] > best_v:
                best_v, best_x = float(vals[i]), X[i].copy()

    def record():
        log.append((gen, ev.calls, best_v if best_x is not None else np.nan,
                    -F.min(), -F.mean(), feas.mean()))

    update_best()
    record()
    while ev.remaining > 0:
        gen += 1
        n_off = min(n - e, ev.remaining)
        n_pairs = -(-n_off // 2)
        par = tournament(rng, F, 2 * n_pairs, g["tournament_k"])
        P1, P2 = X[par[0::2]], X[par[1::2]]
        C1, C2 = CROSSOVERS[g["crossover"]](rng, P1, P2)
        no_cx = rng.random(n_pairs) >= g["p_crossover"]
        C1[no_cx], C2[no_cx] = P1[no_cx], P2[no_cx]
        C = _mutate(rng, np.vstack([C1, C2])[:n_off], g)
        C, Fc, feas_c = evaluate(C)
        keep = elite_indices(F, e)
        X = np.vstack([X[keep], C])
        F = np.concatenate([F[keep], Fc])
        feas = np.concatenate([feas[keep], feas_c])
        update_best()
        if gen % g["log_every"] == 0 or ev.remaining == 0:
            record()
    return {"best_value": best_v if best_x is not None else np.nan, "best_x": best_x,
            "feasible": best_x is not None, "evals": ev.calls, "gens": gen,
            "time_s": time.perf_counter() - t0, "log": log}


def run_random_feasible(cfg, seed, inst=None):
    """Случайный допустимый поиск: случайный порядок приборов + жадная вставка, если не нарушает ограничений.

    Каждое построенное решение — один вызов ФФ; бюджет и порции те же, что у ГА.
    """
    g = cfg["ga"]
    inst = inst or load_instance(cfg["instance"])
    rng = np.random.default_rng(seed)
    rep = Repairer(inst, fill=True)
    ev = Evaluator(lambda X: X @ inst.value, g["budget"])
    t0 = time.perf_counter()
    best_v, best_x, log, gen = -np.inf, None, [], 0
    while ev.remaining > 0:
        k = min(g["pop_size"], ev.remaining)
        X = np.zeros((k, inst.n), dtype=np.int8)
        for r in range(k):
            rep.order_best = list(rng.permutation(inst.n))  # случайный порядок вставки
            X[r] = rep.repair_one(X[r])
        V = ev(X)
        i = int(np.argmax(V))
        if V[i] > best_v:
            best_v, best_x = float(V[i]), X[i].copy()
        if gen % g["log_every"] == 0 or ev.remaining == 0:
            log.append((gen, ev.calls, best_v, V.max(), V.mean(), 1.0))
        gen += 1
    return {"best_value": best_v, "best_x": best_x, "feasible": True, "evals": ev.calls, "gens": gen,
            "time_s": time.perf_counter() - t0, "log": log}


def run_greedy(cfg, seed, inst=None):
    """Конструктивная эвристика: вставка по убыванию v_i / (c_i/B + w_i/P). Детерминирована, 1 вызов ФФ."""
    inst = inst or load_instance(cfg["instance"])
    t0 = time.perf_counter()
    x = Repairer(inst, fill=True).repair_one(np.zeros(inst.n, dtype=np.int8))
    v = float(x @ inst.value)
    return {"best_value": v, "best_x": x, "feasible": True, "evals": 1, "gens": 0,
            "time_s": time.perf_counter() - t0, "log": [(0, 1, v, v, v, 1.0)]}


ALGORITHMS = {"ga": run_ga, "random_feasible": run_random_feasible, "greedy": run_greedy}

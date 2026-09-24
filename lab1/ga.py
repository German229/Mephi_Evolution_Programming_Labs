"""ЛР1: вещественный ГА и случайный поиск с одинаковым учётом бюджета."""
import time

import numpy as np

from evo_core.bounds import reflect
from evo_core.elitism import elite_indices
from evo_core.population import Evaluator, init_uniform
from evo_core.selection import rank_selection, tournament
from lab1.operators import (PROBLEMS, blx_alpha, gaussian_mutation,
                            self_adapt_sigma, uniform_crossover)

LOG_COLUMNS = ["gen", "evals", "best_so_far", "pop_best", "pop_mean", "pop_median", "sigma_med"]


def _problem(cfg):
    p = cfg["problem"]
    d = p["dim"]
    return (PROBLEMS[p["name"]], np.full(d, float(p["lower"])), np.full(d, float(p["upper"])))


def _select(rng, F, n, g):
    if g["selection"] == "tournament":
        return tournament(rng, F, n, g["tournament_k"])
    if g["selection"] == "rank":
        return rank_selection(rng, F, n, g["rank_s"])
    raise ValueError(g["selection"])


def _crossover(rng, P1, P2, g):
    if g["crossover"] == "blx":
        return blx_alpha(rng, P1, P2, g["blx_alpha"]), blx_alpha(rng, P1, P2, g["blx_alpha"])
    if g["crossover"] == "uniform":
        return uniform_crossover(rng, P1, P2)
    raise ValueError(g["crossover"])


def run_ga(cfg, seed):
    g = cfg["ga"]
    func, lower, upper = _problem(cfg)
    width = upper - lower
    rng = np.random.default_rng(seed)
    ev = Evaluator(func, g["budget"])
    n, e = g["pop_size"], g["elite"]
    adaptive = g.get("adaptive", False)
    t0 = time.perf_counter()

    X = init_uniform(rng, n, lower, upper)
    F = ev(X)
    S = np.full(n, float(g["sigma_frac"]))  # шаг мутации в долях ширины области
    best_i = int(np.argmin(F))
    best_f, best_x = float(F[best_i]), X[best_i].copy()
    log, gen = [], 0

    def record():
        log.append((gen, ev.calls, best_f, F.min(), F.mean(), np.median(F), np.median(S)))

    record()
    while ev.remaining > 0:
        gen += 1
        n_off = min(n - e, ev.remaining)
        n_pairs = -(-n_off // 2)
        par = _select(rng, F, 2 * n_pairs, g)
        p1, p2 = par[0::2], par[1::2]
        C1, C2 = _crossover(rng, X[p1], X[p2], g)
        no_cx = rng.random(n_pairs) >= g["p_crossover"]  # пары без кроссовера копируют родителей
        C1[no_cx], C2[no_cx] = X[p1[no_cx]], X[p2[no_cx]]
        C = np.vstack([C1, C2])[:n_off]
        Sc = np.tile(np.sqrt(S[p1] * S[p2]), 2)[:n_off]  # шаг наследуется как среднее геометрическое
        if adaptive:
            Sc = self_adapt_sigma(rng, Sc, g["tau"], g["sigma_min_frac"], g["sigma_frac_max"])
        C = gaussian_mutation(rng, C, Sc[:, None] * width, g["p_mutation"])
        C = reflect(C, lower, upper)
        Fc = ev(C)

        keep = elite_indices(F, e)
        X = np.vstack([X[keep], C])
        F = np.concatenate([F[keep], Fc])
        S = np.concatenate([S[keep], Sc])
        i = int(np.argmin(F))
        if F[i] < best_f:
            best_f, best_x = float(F[i]), X[i].copy()
        if gen % g["log_every"] == 0 or ev.remaining == 0:
            record()

    return {"best_f": best_f, "best_x": best_x, "evals": ev.calls, "gens": gen,
            "time_s": time.perf_counter() - t0,
            "feasible": bool(np.all((best_x >= lower) & (best_x <= upper))), "log": log}


def run_random_search(cfg, seed):
    """Равномерная случайная выборка в боксе; бюджет и порции те же, что у ГА."""
    g = cfg["ga"]
    func, lower, upper = _problem(cfg)
    rng = np.random.default_rng(seed)
    ev = Evaluator(func, g["budget"])
    batch = g["pop_size"]
    t0 = time.perf_counter()
    best_f, best_x, log, gen = np.inf, None, [], 0
    while ev.remaining > 0:
        X = init_uniform(rng, min(batch, ev.remaining), lower, upper)
        F = ev(X)
        i = int(np.argmin(F))
        if F[i] < best_f:
            best_f, best_x = float(F[i]), X[i].copy()
        if gen % g["log_every"] == 0 or ev.remaining == 0:
            log.append((gen, ev.calls, best_f, F.min(), F.mean(), np.median(F), np.nan))
        gen += 1
    return {"best_f": best_f, "best_x": best_x, "evals": ev.calls, "gens": gen,
            "time_s": time.perf_counter() - t0, "feasible": True, "log": log}


ALGORITHMS = {"ga": run_ga, "random": run_random_search}

"""ЛР3: собственная реализация NSGA-II, взвешенная свёртка и случайный поиск для сравнения."""
import time

import numpy as np

from evo_core.elitism import elite_indices
from evo_core.moo import crowded_tournament, hypervolume, nondominated, survival
from evo_core.population import Evaluator, init_uniform
from evo_core.selection import tournament
from lab3.model import load_model
from lab3.operators import polynomial_mutation, sbx

LOG_COLUMNS = ["gen", "evals", "hv", "front_size"]


def setup(cfg):
    model = load_model(cfg["weather"], **cfg.get("model", {}))
    lower, upper = np.zeros(model.days), np.full(model.days, model.i_max)
    nadir = model.nadir()
    return model, lower, upper, nadir


def norm_hv(F, nadir, ref):
    """Гиперобъём в нормированных координатах F/nadir с опорной точкой ref (одинаково для всех методов)."""
    return hypervolume(F / nadir, np.full(F.shape[1], ref))


def variation(rng, X, parents, g, lower, upper):
    n_pairs = len(parents) // 2
    P1, P2 = X[parents[0::2]], X[parents[1::2]]
    C1, C2 = sbx(rng, P1, P2, g["eta_c"], lower, upper)
    no_cx = rng.random(n_pairs) >= g["p_crossover"]
    C1[no_cx], C2[no_cx] = P1[no_cx], P2[no_cx]
    C = np.vstack([C1, C2])
    p_m = g["p_mutation"] if g["p_mutation"] > 0 else 1.0 / X.shape[1]
    return polynomial_mutation(rng, C, g["eta_m"], lower, upper, p_m)


def run_nsga2(cfg, seed):
    g = cfg["moea"]
    model, lower, upper, nadir = setup(cfg)
    rng = np.random.default_rng(seed)
    ev = Evaluator(model.simulate, g["budget"])
    n = g["pop_size"]
    t0 = time.perf_counter()
    X = init_uniform(rng, n, lower, upper)
    F = ev(X)
    keep, ranks, crowd = survival(F, n)
    log, gen = [], 0

    def record():
        front = F[ranks == 0]
        log.append((gen, ev.calls, norm_hv(front, nadir, g["hv_ref"]), len(front)))

    record()
    while ev.remaining > 0:
        gen += 1
        n_off = min(n, ev.remaining)
        par = crowded_tournament(rng, ranks, crowd, 2 * (-(-n_off // 2)))
        C = variation(rng, X, par, g, lower, upper)[:n_off]
        Fc = ev(C)
        XU, FU = np.vstack([X, C]), np.vstack([F, Fc])
        keep, ranks, crowd = survival(FU, n)
        X, F = XU[keep], FU[keep]
        if gen % g["log_every"] == 0 or ev.remaining == 0:
            record()
    idx = np.flatnonzero(ranks == 0)
    idx = idx[nondominated(F[idx])]  # без дубликатов-доминируемых
    return finish(X[idx], F[idx], nadir, g, ev, gen, t0, log)


def finish(X, F, nadir, g, ev, gen, t0, log):
    return {"X": X, "F": F, "hv": norm_hv(F, nadir, g["hv_ref"]), "front_size": len(F), "evals": ev.calls,
            "gens": gen, "time_s": time.perf_counter() - t0, "log": log}


def weight_vectors(h):
    """Равномерная решётка весов на симплексе для 3 критериев (Das—Dennis), C(h+2, 2) векторов."""
    return np.array([(i / h, j / h, (h - i - j) / h) for i in range(h + 1) for j in range(h + 1 - i)])


def run_weighted_sum(cfg, seed):
    """Взвешенная свёртка: для каждого вектора весов свой однокритериальный ГА на части общего бюджета.

    Минимизируется Σ w_k · f_k / nadir_k; те же SBX и полиномиальная мутация, турнир и элитизм из ядра.
    """
    g = cfg["moea"]
    model, lower, upper, nadir = setup(cfg)
    rng = np.random.default_rng(seed)
    W = weight_vectors(g["ws_h"])
    ev = Evaluator(model.simulate, g["budget"])
    share = g["budget"] // len(W)
    n = g["ws_pop_size"]
    t0 = time.perf_counter()
    XS, FS, log = [], [], []
    for k, w in enumerate(W):
        stop = ev.calls + share if k < len(W) - 1 else g["budget"]
        X = init_uniform(rng, n, lower, upper)
        FX = ev(X)
        S = (FX / nadir) @ w
        while ev.calls < stop:
            n_off = min(n - g["elite"], stop - ev.calls)
            par = tournament(rng, S, 2 * (-(-n_off // 2)), g["tournament_k"])
            C = variation(rng, X, par, g, lower, upper)[:n_off]
            FC = ev(C)
            keep = elite_indices(S, g["elite"])
            X, FX = np.vstack([X[keep], C]), np.vstack([FX[keep], FC])
            S = (FX / nadir) @ w
        b = int(np.argmin(S))
        XS.append(X[b])
        FS.append(FX[b])
        FA = np.array(FS)
        log.append((k + 1, ev.calls, norm_hv(FA[nondominated(FA)], nadir, g["hv_ref"]), len(nondominated(FA))))
    XS, FS = np.array(XS), np.array(FS)
    idx = nondominated(FS)
    return finish(XS[idx], FS[idx], nadir, g, ev, len(W), t0, log)


def run_random(cfg, seed):
    """Случайный поиск: равномерные расписания, внешний архив недоминируемых, тот же бюджет."""
    g = cfg["moea"]
    model, lower, upper, nadir = setup(cfg)
    rng = np.random.default_rng(seed)
    ev = Evaluator(model.simulate, g["budget"])
    t0 = time.perf_counter()
    AX, AF = np.empty((0, model.days)), np.empty((0, 3))
    log, gen = [], 0
    while ev.remaining > 0:
        X = init_uniform(rng, min(g["pop_size"], ev.remaining), lower, upper)
        AX, AF = np.vstack([AX, X]), np.vstack([AF, ev(X)])
        idx = nondominated(AF)
        AX, AF = AX[idx], AF[idx]
        gen += 1
        if gen % g["log_every"] == 0 or ev.remaining == 0:
            log.append((gen, ev.calls, norm_hv(AF, nadir, g["hv_ref"]), len(AF)))
    return finish(AX, AF, nadir, g, ev, gen, t0, log)


ALGORITHMS = {"nsga2": run_nsga2, "weighted_sum": run_weighted_sum, "random": run_random}

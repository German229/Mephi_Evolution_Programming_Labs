"""ЛР4: генетическое программирование для символьной регрессии с четырьмя режимами контроля bloat.

A  "depth"     — только предел глубины/размера (эталон);
B  "parsimony" — фитнес NRMSE + c·size;
C  "tarpeian"  — особь крупнее среднего с вероятностью p получает худший фитнес без вычисления (Poli, 2003);
D  "nsga2"     — два критерия (NRMSE, size), собственный NSGA-II из evo_core.moo.
Во всех вариантах одинаковы язык, инициализация, операторы и бюджет вычислений ФФ;
итоговая модель выбирается одинаково — минимум NRMSE на validation среди финальной популяции (для D — первого фронта).
"""
import time

import numpy as np

from evo_core.elitism import elite_indices
from evo_core.moo import crowded_tournament, survival
from evo_core.selection import tournament
from lab4 import data
from lab4.tree import (Language, depth, point_mutation, predict, subtree_crossover, subtree_mutation,
                       to_code, to_infix)

BAD = 1e3  # «худший» фитнес для неконечных предсказаний
LOG_COLUMNS = ["gen", "evals", "best_train", "best_size", "mean_size", "median_size", "nodes"]


class Fitness:
    """Линейно масштабированная NRMSE (Keijzer, 2003): a, b подбираются на train МНК.

    Кэш по строковому коду дерева хранит NRMSE на всех разбиениях; каждый вызов считается вызовом ФФ
    (бюджет — логический), а в nodes добавляется размер дерева (стоимость вычисления).
    """

    def __init__(self, X, y, masks, budget):
        self.X, self.y, self.m = X, y, masks
        self.scale = y[masks["train"]].std()
        self.budget, self.calls, self.nodes = budget, 0, 0
        self.cache = {}

    @property
    def remaining(self):
        return self.budget - self.calls

    def errors(self, tree):
        key = to_code(tree)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        p = predict(tree, self.X)
        tr = self.m["train"]
        if not np.all(np.isfinite(p)):
            res = (BAD, BAD, BAD, BAD, 0.0, 0.0)
        else:
            pt, yt = p[tr], self.y[tr]
            var = pt.var()
            b = 0.0 if var < 1e-12 else float(np.cov(pt, yt, bias=True)[0, 1] / var)
            a = float(yt.mean() - b * pt.mean())
            q = a + b * p
            res = tuple(min(BAD, float(np.sqrt(np.mean((q[self.m[s]] - self.y[self.m[s]]) ** 2)) / self.scale))
                        for s in ("train", "val", "test", "extrap")) + (a, b)
        if len(self.cache) > 50000:
            self.cache.clear()
        self.cache[key] = res
        return res

    def __call__(self, trees):
        if self.calls + len(trees) > self.budget:
            raise RuntimeError("бюджет вычислений ФФ исчерпан")
        self.calls += len(trees)
        self.nodes += sum(len(t) for t in trees)
        return np.array([self.errors(t)[0] for t in trees])


def _breed(rng, pop, parents, lang, g):
    kids = []
    for k in range(0, len(parents), 2):
        p1, p2 = pop[parents[k]], pop[parents[k + 1]]
        r = rng.random()
        if r < g["p_crossover"]:
            kids.extend(subtree_crossover(rng, p1, p2, g["max_depth"], g["max_size"]))
        elif r < g["p_crossover"] + g["p_subtree_mutation"]:
            kids += [subtree_mutation(rng, p1, lang, g["max_depth"], g["max_size"]),
                     subtree_mutation(rng, p2, lang, g["max_depth"], g["max_size"])]
        else:
            kids += [point_mutation(rng, p1, lang), point_mutation(rng, p2, lang)]
    return kids


def run_gp(cfg, seed, problem):
    g = cfg["gp"]
    mode = cfg["bloat_control"]
    X, y, masks = data.load(problem)
    rng = np.random.default_rng(seed)
    lang = Language(g["functions"], X.shape[1], g["p_const"])
    fit = Fitness(X, y, masks, g["budget"])
    n = g["pop_size"]
    t0 = time.perf_counter()

    pop = lang.ramped_half_and_half(rng, n, g["init_min_depth"], g["init_max_depth"])
    err = fit(pop)
    size = np.array([len(t) for t in pop], float)

    def scalar(err, size):
        return err + g.get("parsimony_c", 0.0) * size if mode == "parsimony" else err

    if mode == "nsga2":
        keep, ranks, crowd = survival(np.column_stack([err, size]), n)
        pop, err, size = [pop[i] for i in keep], err[keep], size[keep]
    log, gen = [], 0

    def record():
        i = int(np.argmin(err))
        log.append((gen, fit.calls, err[i], size[i], size.mean(), np.median(size), fit.nodes))

    record()
    while fit.remaining > 0:
        gen += 1
        if mode == "nsga2":
            n_off = min(n, fit.remaining)
            par = crowded_tournament(rng, ranks, crowd, 2 * (-(-n_off // 2)))
            kids = _breed(rng, pop, par, lang, g)[:n_off]
            ek, sk = fit(kids), np.array([len(t) for t in kids], float)
            U = pop + kids
            EU, SU = np.concatenate([err, ek]), np.concatenate([size, sk])
            keep, ranks, crowd = survival(np.column_stack([EU, SU]), n)
            pop, err, size = [U[i] for i in keep], EU[keep], SU[keep]
        else:
            e = g["elite"]
            f = scalar(err, size)
            par = tournament(rng, f, 2 * (-(-(n - e) // 2)), g["tournament_k"])
            kids = _breed(rng, pop, par, lang, g)[: n - e]
            sk = np.array([len(t) for t in kids], float)
            ek = np.full(len(kids), BAD)
            if mode == "tarpeian":
                # «убитые» крупные особи не вычисляются — экономия бюджета ФФ
                alive = ~((sk > size.mean()) & (rng.random(len(kids)) < g["tarpeian_p"]))
            else:
                alive = np.ones(len(kids), bool)
            idx = np.flatnonzero(alive)[: fit.remaining]
            alive[:] = False
            alive[idx] = True
            ek[alive] = fit([kids[i] for i in idx])
            keep = elite_indices(f, e)
            pop = [pop[i] for i in keep] + kids
            err = np.concatenate([err[keep], ek])
            size = np.concatenate([size[keep], sk])
        if gen % g["log_every"] == 0 or fit.remaining == 0:
            record()

    # итоговая модель: минимум validation NRMSE (для D — среди первого фронта)
    cand = np.flatnonzero(ranks == 0) if mode == "nsga2" else np.flatnonzero(err < BAD)
    if cand.size == 0:
        cand = np.arange(len(pop))
    errs = [fit.errors(pop[i]) for i in cand]
    best = int(cand[int(np.argmin([e_[1] for e_ in errs]))])
    tr, va, te, ex, a, b = fit.errors(pop[best])
    out = {"tree": pop[best], "train": tr, "val": va, "test": te, "extrap": ex, "a": a, "b": b,
           "size": len(pop[best]), "depth": depth(pop[best]), "evals": fit.calls, "gens": gen,
           "nodes": fit.nodes, "time_s": time.perf_counter() - t0, "log": log, "expr": to_infix(pop[best])}
    if mode == "nsga2":  # фронт (ошибка на train, размер) с validation — для анализа компромисса
        f0 = np.flatnonzero(ranks == 0)
        out["front"] = sorted({(int(size[i]), float(err[i]), float(fit.errors(pop[i])[1]), to_infix(pop[i]))
                               for i in f0})
    return out

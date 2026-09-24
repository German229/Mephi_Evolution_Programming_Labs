"""ЛР4: статистика серий (медианы, U-тест с поправкой Холма, Â12), динамика bloat, формулы, графики.

Пример: python -m lab4.analyze --plots
Для графиков аппроксимации медианные запуски пересчитываются заново по seed — заодно проверяется воспроизводимость.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evo_core import plotting
from evo_core.plotting import INK, INK_2, SERIES, SURFACE, plt
from evo_core.stats import holm, mann_whitney, read_csv, vargha_delaney_a12
from lab4 import data
from lab4.gp import run_gp
from lab4.tree import predict

RESULTS = Path(__file__).parent / "results"
VARIANTS = ["gp_depth", "gp_parsimony", "gp_tarpeian", "gp_nsga2"]
LABELS = {"gp_depth": "A: предел глубины", "gp_parsimony": "B: парсимония",
          "gp_tarpeian": "C: тарпейский", "gp_nsga2": "D: NSGA-II"}
PROBLEMS = ["nguyen7", "pagie1"]
NUM = ["train", "val", "test", "extrap", "size", "depth", "evals", "gens", "nodes", "time_s"]


def summary(problem, v):
    rows = read_csv(RESULTS / f"{problem}_{v}_summary.csv")
    return {k: np.array([float(r[k]) for r in rows]) for k in NUM} | {"expr": [r["expr"] for r in rows],
                                                                      "seed": [int(r["seed"]) for r in rows]}


def curves(problem, v, col):
    by_run, xs = defaultdict(list), defaultdict(list)
    for r in read_csv(RESULTS / f"{problem}_{v}_gens.csv"):
        by_run[r["run"]].append(float(r[col]))
        xs[r["run"]].append(float(r["evals"]))
    grid = np.linspace(0, 20000, 101)  # у C больше поколений, поэтому кривые выравниваются по вызовам ФФ
    return grid, np.array([np.interp(grid, xs[k], by_run[k]) for k in by_run])


def iqr(v):
    return f"{np.percentile(v, 25):.4f}–{np.percentile(v, 75):.4f}"


def table(problem):
    lines = [f"### {problem}", "",
             "| вариант | test NRMSE: медиана | IQR | best | mean | std | worst | extrap NRMSE (мед.) | "
             "размер (мед.) | глубина (мед.) | поколений | вычислено узлов, млн (мед.) | время/запуск, с |",
             "|" + "---|" * 13]
    for v in VARIANTS:
        s = summary(problem, v)
        t = s["test"]
        lines.append(f"| {LABELS[v]} | {np.median(t):.4f} | {iqr(t)} | {t.min():.4f} | {t.mean():.4f} | "
                     f"{t.std(ddof=1):.4f} | {t.max():.4f} | {np.median(s['extrap']):.3f} | {np.median(s['size']):.0f} | "
                     f"{np.median(s['depth']):.0f} | {np.median(s['gens']):.0f} | {np.median(s['nodes']) / 1e6:.2f} | "
                     f"{s['time_s'].mean():.2f} |")
    return lines


def tests(problem):
    """Каждый вариант против A по test NRMSE и размеру; поправка Холма внутри задачи и метрики."""
    out = [f"### {problem}", "", "| сравнение | метрика | медиана варианта | медиана A | z | p (Холм) | Â12 (вариант лучше A) |",
           "|---|---|---|---|---|---|---|"]
    ref = summary(problem, "gp_depth")
    for metric in ("test", "size"):
        rows, ps = [], []
        for v in VARIANTS[1:]:
            s = summary(problem, v)
            _, z, p = mann_whitney(s[metric], ref[metric])
            rows.append((v, np.median(s[metric]), np.median(ref[metric]), z,
                         vargha_delaney_a12(ref[metric], s[metric])))  # P(A хуже варианта)
            ps.append(p)
        for (v, mv, mr, z, a12), p in zip(rows, holm(ps)):
            fm = "{:.4f}" if metric == "test" else "{:.0f}"
            out.append(f"| {LABELS[v]} vs A | {metric} | {fm.format(mv)} | {fm.format(mr)} | {z:.2f} | {p:.2e} | {a12:.2f} |")
    # дополнительная пара: два лучших механизма между собой
    b, d = summary(problem, "gp_parsimony"), summary(problem, "gp_nsga2")
    _, z, p = mann_whitney(d["test"], b["test"])
    out.append(f"| D vs B (без поправки) | test | {np.median(d['test']):.4f} | {np.median(b['test']):.4f} | {z:.2f} | "
               f"{p:.2e} | {vargha_delaney_a12(b['test'], d['test']):.2f} |")
    return out


def median_run(problem, v):
    s = summary(problem, v)
    k = int(np.argsort(s["test"])[len(s["test"]) // 2])
    return k, s["seed"][k], s


def formulas(problem):
    lines = [f"### {problem}: итоговые формулы медианных по test запусков (+ линейное масштабирование a + b·h)", "",
             "| вариант | seed | размер | test NRMSE | формула h(x) |", "|---|---|---|---|---|"]
    for v in VARIANTS:
        k, seed, s = median_run(problem, v)
        e = s["expr"][k]
        lines.append(f"| {LABELS[v]} | {seed} | {s['size'][k]:.0f} | {s['test'][k]:.4f} | "
                     f"`{e if len(e) <= 160 else e[:157] + '…'}` |")
    fr = read_csv(RESULTS / f"{problem}_gp_nsga2_fronts.csv")
    k, seed, _ = median_run(problem, "gp_nsga2")
    front = sorted([r for r in fr if int(r["seed"]) == seed], key=lambda r: int(r["size"]))
    lines += ["", f"Фронт Парето (NRMSE train, размер) варианта D, запуск seed {seed}: "
                  "три решения разной сложности", "",
              "| размер | NRMSE train | NRMSE val | формула |", "|---|---|---|---|"]
    picks = {0, len(front) // 2, len(front) - 1}
    for i in sorted(picks):
        r = front[i]
        lines.append(f"| {r['size']} | {float(r['train']):.4f} | {float(r['val']):.4f} | `{r['expr'][:140]}` |")
    return lines


# ---------- графики ----------
def plot_dynamics(problem):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
    for v, c in zip(VARIANTS, SERIES):
        x, M = curves(problem, v, "mean_size")
        plotting.band_plot(a1, x, M, c, LABELS[v], fmt="{:.0f}")
        x, M = curves(problem, v, "best_train")
        plotting.band_plot(a2, x, M, c, LABELS[v], fmt="{:.3f}")
    plotting._style(a1, "вычисления ФФ", "средний размер дерева в популяции, узлов")
    plotting._style(a2, "вычисления ФФ", "лучшая NRMSE на train")
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h[1::2], [x.replace(": медиана", "") for x in lab[1::2]], frameon=False, fontsize=9,
               labelcolor=INK, ncol=4, loc="lower center")
    a1.set_title(f"{problem}: разрастание (слева) и ошибка (справа), медиана и IQR по 30 запускам",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(RESULTS / f"dynamics_{problem}.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def plot_error_vs_size():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=SURFACE)
    for ax, problem in zip(axes, PROBLEMS):
        for v, c, mk in zip(VARIANTS, SERIES, "osD^"):
            s = summary(problem, v)
            ax.scatter(s["size"], s["test"], s=26, marker=mk, color=c, alpha=0.8, edgecolors=SURFACE,
                       linewidths=0.8, label=LABELS[v], zorder=3)
        ax.set_xscale("log")
        plotting._style(ax, "размер итоговой модели, узлов", "NRMSE на test")
        ax.set_title(problem, loc="left", color=INK, fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "error_vs_size.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def plot_fronts():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=SURFACE)
    for ax, problem in zip(axes, PROBLEMS):
        k, seed, _ = median_run(problem, "gp_nsga2")
        fr = sorted([r for r in read_csv(RESULTS / f"{problem}_gp_nsga2_fronts.csv") if int(r["seed"]) == seed],
                    key=lambda r: int(r["size"]))
        s = [int(r["size"]) for r in fr]
        ax.step(s, [float(r["train"]) for r in fr], where="post", color=SERIES[0], linewidth=2, label="train")
        ax.plot(s, [float(r["val"]) for r in fr], "o", color=SERIES[1], markersize=5, label="validation")
        plotting._style(ax, "размер дерева, узлов", "NRMSE")
        ax.set_title(f"{problem}: фронт Парето варианта D (seed {seed})", loc="left", color=INK, fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "nsga2_fronts.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def rerun(problem, v):
    cfg = json.loads((RESULTS / f"{problem}_{v}_config.json").read_text())
    k, seed, s = median_run(problem, v)
    r = run_gp(cfg, seed, problem)
    assert np.isclose(r["test"], s["test"][k], rtol=1e-4), "запуск не воспроизвёлся"
    return r


def plot_fits(models):
    X, y, m = data.load("nguyen7")
    grid = np.linspace(0, 3, 300)[:, None]
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    ax.axvspan(2, 3, color=INK_2, alpha=0.07, linewidth=0)
    ax.annotate("экстраполяция", (2.05, 0.1), color=INK_2, fontsize=9)
    ax.scatter(X[m["train"], 0], y[m["train"]], s=10, color=INK_2, alpha=0.5, label="train (с шумом)")
    ax.plot(grid[:, 0], data.nguyen7(grid), color=INK, linewidth=2.5, linestyle="--", label="истинная функция")
    for (v, r), c in zip(models["nguyen7"].items(), SERIES):
        ax.plot(grid[:, 0], r["a"] + r["b"] * predict(r["tree"], grid), color=c, linewidth=1.6,
                label=f"{LABELS[v]} (размер {r['size']})")
    ax.set_ylim(-0.5, 5)
    plotting._style(ax, "x", "y", log=False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    ax.set_title("Nguyen-7: модели медианных запусков", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "fit_nguyen7.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)

    X, y, m = data.load("pagie1")
    sel = m["test"] | m["extrap"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), facecolor=SURFACE, sharey=True)
    for ax, (v, r), c in zip(axes, models["pagie1"].items(), SERIES):
        p = r["a"] + r["b"] * predict(r["tree"], X[sel])
        ax.plot([0, 2], [0, 2], color=INK_2, linewidth=1, linestyle="--")
        ax.scatter(y[m["test"]], p[m["test"][sel]], s=12, color=c, alpha=0.8, label="test")
        ax.scatter(y[m["extrap"]], p[m["extrap"][sel]], s=12, marker="x", color=INK_2, label="экстраполяция")
        ax.set_ylim(-0.5, 2.5)
        plotting._style(ax, "истинное y", "предсказание" if ax is axes[0] else "", log=False)
        ax.set_title(f"{LABELS[v]} · размер {r['size']}", loc="left", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "fit_pagie1.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()
    md = ["# ЛР4: сводка серий (30 запусков на вариант и задачу, 20 000 вычислений ФФ)", "",
          "## Качество и размер итоговых моделей", ""]
    for pr in PROBLEMS:
        md += table(pr) + [""]
    md += ["## Сравнение с базовым вариантом A: U-тест Манна—Уитни, поправка Холма, размер эффекта Â12", "",
           "z < 0 — у варианта значения меньше (лучше). Â12 > 0.5 — вариант лучше A; 0.56/0.64/0.71 — малый/средний/большой эффект.", ""]
    for pr in PROBLEMS:
        md += tests(pr) + [""]
    md += ["## Формулы", ""]
    for pr in PROBLEMS:
        md += formulas(pr) + [""]
    (RESULTS / "summary.md").write_text("\n".join(md))
    for pr in PROBLEMS:
        for v in VARIANTS:
            s = summary(pr, v)
            print(f"{pr:<8} {LABELS[v]:<20} test med={np.median(s['test']):.4f} size med={np.median(s['size']):.0f} "
                  f"nodes={np.median(s['nodes']) / 1e6:.2f}M t={s['time_s'].mean():.1f}s")
    if args.plots:
        for pr in PROBLEMS:
            plot_dynamics(pr)
        plot_error_vs_size()
        plot_fronts()
        plot_fits({pr: {v: rerun(pr, v) for v in VARIANTS} for pr in PROBLEMS})


if __name__ == "__main__":
    main()

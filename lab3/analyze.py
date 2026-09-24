"""ЛР3: гиперобъём серий, U-тесты, проекции фронта Парето, характерные решения.

Пример: python -m lab3.analyze --plots
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from evo_core import plotting
from evo_core.plotting import INK, INK_2, SERIES, SURFACE, plt
from evo_core.stats import describe, mann_whitney, read_csv
from evo_core.moo import nondominated
from lab3.model import OBJECTIVES, load_model

RESULTS = Path(__file__).parent / "results"
ORDER = ["nsga2", "nsga2_eta5", "weighted_sum", "weighted_sum_10x", "random_search"]
LABELS = {"nsga2": "NSGA-II, η_m=20", "nsga2_eta5": "NSGA-II, η_m=5", "weighted_sum": "взвеш. свёртка",
          "weighted_sum_10x": "взвеш. свёртка, 10× бюджет", "random_search": "случайный поиск"}
TESTS = [("nsga2", "weighted_sum"), ("nsga2", "weighted_sum_10x"), ("nsga2", "random_search"),
         ("nsga2_eta5", "nsga2")]
DEFICIT_TOL = 0.05  # режим «малый стресс»: дефицит не более 5 % от дефицита без полива


def summary(tag):
    rows = read_csv(RESULTS / f"{tag}_summary.csv")
    return {k: np.array([float(r[k]) for r in rows]) for k in ("hv", "front_size", "evals", "time_s")}


def curves(tag):
    by_run, xs = defaultdict(list), defaultdict(list)
    for r in read_csv(RESULTS / f"{tag}_gens.csv"):
        by_run[r["run"]].append(float(r["hv"]))
        xs[r["run"]].append(float(r["evals"]))
    return np.array(next(iter(xs.values()))), np.array(list(by_run.values()))


def median_front(tag):
    rows = read_csv(RESULTS / f"{tag}_median_front.csv")
    F = np.array([[float(r[k]) for k in ("water", "deficit", "wet")] for r in rows])
    X = np.array([[float(v) for k, v in r.items() if k.startswith("I")] for r in rows])
    return F, X


def characteristic(F, nadir):
    """Индексы характерных решений фронта (в нормированных координатах f/nadir)."""
    Fn = F / nadir
    ideal = Fn.min(0)
    span = np.maximum(Fn.max(0) - ideal, 1e-12)
    ok = np.flatnonzero(F[:, 1] <= DEFICIT_TOL * nadir[1])
    return {
        "минимум воды": int(np.lexsort((F[:, 1], F[:, 0]))[0]),                 # самое дешёвое решение фронта
        "сбалансированный (колено)": int(np.argmin(np.linalg.norm((Fn - ideal) / span, axis=1))),
        "малый стресс": int(ok[np.argmin(F[ok, 0])]),                            # min вода при дефиците ≤ 5 %
        "без стресса": int(np.lexsort((F[:, 0], F[:, 1]))[0]),                  # min дефицит, затем min вода
    }


def table(tags):
    lines = ["| метод | HV best | HV mean | HV median | HV std | HV worst | точек фронта (медиана) | "
             "время/запуск, с | вызовов модели |", "|" + "---|" * 9]
    for t in tags:
        s = summary(t)
        d = describe(s["hv"])
        lines.append(f"| {LABELS[t]} | {d['worst']:.4f} | {d['mean']:.4f} | {d['median']:.4f} | {d['std']:.4f} | "
                     f"{d["best"]:.4f} | {np.median(s['front_size']):.0f} | {s['time_s'].mean():.3f} | "
                     f"{int(s['evals'].mean())} |")
    return lines  # HV максимизируется: «best» — максимум (describe считает min как best)


def tests(tags):
    lines = ["| A vs B | HV медиана A | HV медиана B | U | z | p |", "|---|---|---|---|---|---|"]
    for a, b in TESTS:
        if a in tags and b in tags:
            ha, hb = summary(a)["hv"], summary(b)["hv"]
            u, z, p = mann_whitney(ha, hb)
            lines.append(f"| {LABELS[a]} vs {LABELS[b]} | {np.median(ha):.4f} | {np.median(hb):.4f} | "
                         f"{u:.0f} | {z:.2f} | {p:.2e} |")
    return lines


def char_table(F, idx):
    lines = ["| решение | вода, мм | дефицит, мм·сут | переувлажнение, мм·сут | поливов (>0.5 мм) |",
             "|---|---|---|---|---|"]
    for name, i in idx.items():
        lines.append(f"| {name} | {F[i, 0]:.1f} | {F[i, 1]:.1f} | {F[i, 2]:.1f} | — |")
    return lines


def plot_projections(model, nadir, idx, path):
    F, _ = median_front("nsga2")
    WS = [np.array([[float(r[k]) for k in ("water", "deficit", "wet")] for r in read_csv(RESULTS / f"{t}_fronts.csv")])
          for t in ("weighted_sum", "weighted_sum_10x")]
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=SURFACE)
    for ax, (a, b) in zip(axes, pairs):
        ax.scatter(F[:, a], F[:, b], s=16, color=SERIES[0], alpha=0.8, edgecolors=SURFACE, linewidths=0.6,
                   label="NSGA-II (медианный запуск)", zorder=3)
        ax.scatter(WS[0][:, a], WS[0][:, b], s=46, marker="s", color=SERIES[1], edgecolors=SURFACE,
                   linewidths=1, label="взвеш. свёртка (все 20 запусков)", zorder=4)
        ax.scatter(WS[1][:, a], WS[1][:, b], s=46, marker="^", color=SERIES[2], edgecolors=SURFACE,
                   linewidths=1, label="взвеш. свёртка 10× (все 20 запусков)", zorder=4)
        for name, i in idx.items():
            ax.scatter(F[i, a], F[i, b], s=90, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=5)
            ax.annotate(name.replace(" (колено)", ""), (F[i, a], F[i, b]), xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color=INK, annotation_clip=False, zorder=10,
                        bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85})
        ax.set_xscale("symlog", linthresh=10)
        ax.set_yscale("symlog", linthresh=10)
        plotting._style(ax, OBJECTIVES[a], OBJECTIVES[b], log=False)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    axes[0].set_title("Проекции фронта Парето (симлог-оси)", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def plot_schedules(model, F, X, idx, path):
    names = list(idx)
    fig, axes = plt.subplots(2, len(names), figsize=(4.6 * len(names), 6), facecolor=SURFACE, sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
    days = np.arange(1, model.days + 1)
    for col, name in enumerate(names):
        i = idx[name]
        _, tr = model.simulate(X[i][None], trace=True)
        top, bot = axes[0, col], axes[1, col]
        top.axhspan(0, model.theta_low, color=SERIES[1], alpha=0.08, linewidth=0)
        top.axhspan(model.fc, model.theta_sat, color=SERIES[0], alpha=0.08, linewidth=0)
        th = tr["theta"][0]                                  # (K, days)
        top.fill_between(days, th.min(0), th.max(0), color=SERIES[0], alpha=0.25, linewidth=0,
                         label="диапазон по 10 сценариям")
        top.plot(days, th.mean(0), color=INK, linewidth=2, label="среднее")
        top.set_ylim(0, model.theta_sat)
        top.set_title(f"{name}\nвода {F[i, 0]:.0f} мм · дефицит {F[i, 1]:.0f} · переувл. {F[i, 2]:.0f}",
                      loc="left", color=INK, fontsize=10)
        plotting._style(top, "", "запас влаги θ, мм" if col == 0 else "", log=False)
        bot.bar(days - 0.2, X[i], width=0.4, color=SERIES[0], label="полив")
        bot.bar(days + 0.2, model.rain.mean(0), width=0.4, color=SERIES[2], label="осадки (среднее)")
        plotting._style(bot, "сутки", "мм/сут" if col == 0 else "", log=False)
    axes[0, 0].annotate("зона стресса θ < θ_low", (1, 4), color=INK_2, fontsize=8)
    axes[0, 0].annotate("переувлажнение θ > FC", (1, model.fc + 4), color=INK_2, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    axes[0, -1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()
    model = load_model()
    nadir = model.nadir()
    tags = [t for t in ORDER if (RESULTS / f"{t}_summary.csv").exists()]
    F, X = median_front("nsga2")
    idx = characteristic(F, nadir)
    ct = char_table(F, idx)
    for k, (name, i) in enumerate(idx.items()):  # число поливов
        ct[2 + k] = ct[2 + k].replace("| — |", f"| {int((X[i] > 0.5).sum())} |")
    t1, t2 = table(tags), tests(tags)
    (RESULTS / "summary.md").write_text("\n".join([
        f"Нормировка: nadir = {np.round(nadir, 1).tolist()} (полив по максимуму / без полива); "
        f"опорная точка HV = 1.1 по каждой оси, максимум HV = {1.1 ** 3:.3f}.", "",
        "## Гиперобъём (больше — лучше)", "", *t1, "", "## U-тест Манна—Уитни по HV (z > 0: A лучше)", "", *t2, "",
        "## Характерные решения (NSGA-II, запуск с медианным HV)", "", *ct, ""]))
    for line in t1[2:]:
        c = [x.strip() for x in line.split("|")[1:-1]]
        print(f"{c[0]:<28} HV med={c[3]} [{c[5]}..{c[1]}] front={c[6]}")
    for line in ct[2:]:
        print("  " + line)
    if args.plots:
        plotting.compare([(LABELS[t], *curves(t)) for t in ("nsga2", "nsga2_eta5", "random_search") if t in tags],
                         "Гиперобъём по ходу поиска", RESULTS / "hv_convergence.png", log=False, fmt="{:.3f}",
                         ylabel="гиперобъём (норм.)", ref=(1.1 ** 3, "максимум 1.331"))
        plotting.final_boxes([(LABELS[t].replace(", ", "\n"), summary(t)["hv"]) for t in tags],
                             "Итоговый гиперобъём 20 запусков", RESULTS / "hv_boxes.png",
                             ylabel="гиперобъём (норм.)", log=False)
        plot_projections(model, nadir, idx, RESULTS / "front_projections.png")
        plot_schedules(model, F, X, idx, RESULTS / "schedules.png")


if __name__ == "__main__":
    main()

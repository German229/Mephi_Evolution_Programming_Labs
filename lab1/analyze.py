"""ЛР1: сводка серий из results/*.csv, U-тесты Манна—Уитни и графики.

Пример: python -m lab1.analyze --plots
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from evo_core import plotting
from evo_core.stats import describe, mann_whitney, read_csv

RESULTS = Path(__file__).parent / "results"
ORDER = ["ga_base", "ga_sigma_large", "ga_sigma_small", "ga_adaptive", "ga_adaptive_pm03", "random_search"]
LABELS = {"ga_base": "ГА, σ=0.01", "ga_sigma_large": "ГА, σ=0.1", "ga_sigma_small": "ГА, σ=0.001", "ga_adaptive": "ГА, адапт. σ",
          "ga_adaptive_pm03": "ГА, адапт. σ, p_m=0.3", "random_search": "случайный поиск"}
TESTS = [("ga_base", "random_search"), ("ga_base", "ga_sigma_large"),
         ("ga_adaptive", "ga_base"), ("ga_adaptive", "ga_sigma_small"), ("ga_adaptive_pm03", "ga_adaptive")]
HIT = 1e-8  # порог «достигнут глобальный минимум»


def load_summary(tag):
    rows = read_csv(RESULTS / f"{tag}_summary.csv")
    return {k: np.array([float(r[k]) for r in rows]) for k in ("best_f", "evals", "time_s", "feasible")}


def load_curves(tag, xcol="evals", ycol="best_so_far"):
    by_run = defaultdict(list)
    xs = defaultdict(list)
    for r in read_csv(RESULTS / f"{tag}_gens.csv"):
        by_run[r["run"]].append(float(r[ycol]))
        xs[r["run"]].append(float(r[xcol]))
    M = np.array(list(by_run.values()))
    return np.array(next(iter(xs.values()))), M


def table(tags):
    head = ("| конфигурация | best | mean | median | std | worst | f<1e-8 | допустимо | "
            "время/запуск, с | вызовов ФФ |")
    lines = [head, "|" + "---|" * 10]
    for t in tags:
        s = load_summary(t)
        d = describe(s["best_f"])
        lines.append(f"| {LABELS.get(t, t)} | {d['best']:.3e} | {d['mean']:.3e} | {d['median']:.3e} | "
                     f"{d['std']:.3e} | {d['worst']:.3e} | {int((s['best_f'] < HIT).sum())}/{s['best_f'].size} | "
                     f"{100 * s['feasible'].mean():.0f}% | {s['time_s'].mean():.3f} | {int(s['evals'].mean())} |")
    return lines


def tests(tags):
    lines = ["| A vs B | медиана A | медиана B | U | z | p |", "|---|---|---|---|---|---|"]
    for a, b in TESTS:
        if a in tags and b in tags:
            fa, fb = load_summary(a)["best_f"], load_summary(b)["best_f"]
            u, z, p = mann_whitney(fa, fb)
            lines.append(f"| {LABELS[a]} vs {LABELS[b]} | {np.median(fa):.3e} | {np.median(fb):.3e} | "
                         f"{u:.0f} | {z:.2f} | {p:.2e} |")
    return lines


def plots(tags):
    out = RESULTS
    if "ga_base" in tags:
        x, M = load_curves("ga_base", xcol="gen")
        plotting.convergence(x, M, "ГА на функции Гриванка (d=10): 20 запусков", out / "conv_ga_base.png")
    pairs = [("cmp_sigma.png", ["ga_base", "ga_sigma_large"], "Масштаб мутации: σ=0.01 vs σ=0.1 ширины"),
             ("cmp_random.png", ["ga_base", "random_search"], "ГА vs случайный поиск при равном бюджете")]
    for fname, group, title in pairs:
        if all(t in tags for t in group):
            plotting.compare([(LABELS[t], *load_curves(t)) for t in group], title, out / fname)
    group = [t for t in ("ga_base", "ga_adaptive", "ga_adaptive_pm03") if t in tags]
    if len(group) > 1:
        plotting.compare_with_panel(
            [(LABELS[t], *load_curves(t)) for t in group],
            [(LABELS[t], *load_curves(t, ycol="sigma_med")) for t in group],
            "Фиксированный vs самоадаптивный шаг мутации", "σ / ширина области", out / "cmp_adaptive.png")
    plotting.final_boxes([(LABELS[t], load_summary(t)["best_f"]) for t in tags],
                         "Итог 20 запусков по конфигурациям", out / "final_boxes.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--md", default=str(RESULTS / "summary.md"), help="куда записать таблицы")
    args = p.parse_args()
    tags = [t for t in ORDER if (RESULTS / f"{t}_summary.csv").exists()]
    t1, t2 = table(tags), tests(tags)
    Path(args.md).write_text("\n".join(["## Сводка серий", "", *t1, "", "## U-тест Манна—Уитни", "", *t2, ""]))
    for t in tags:  # компактный вывод в терминал
        d = describe(load_summary(t)["best_f"])
        print(f"{t:<17} best={d['best']:.2e} med={d['median']:.2e} mean={d['mean']:.2e} worst={d['worst']:.2e}")
    for line in t2[2:]:
        print(line)
    if args.plots:
        plots(tags)


if __name__ == "__main__":
    main()

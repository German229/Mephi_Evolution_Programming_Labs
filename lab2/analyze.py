"""ЛР2: сводка серий, отклонение от точного оптимума, U-тесты и графики.

Пример: python -m lab2.analyze --plots   (нужен results/exact.json: python -m lab2.exact)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evo_core import plotting
from evo_core.stats import mann_whitney, read_csv

RESULTS = Path(__file__).parent / "results"
ORDER = ["ga_repair", "ga_repair_swap", "ga_repair_nofill", "ga_penalty", "ga_penalty_swap", "ga_penalty_low",
         "random_feasible", "greedy"]
LABELS = {"ga_repair": "ГА: ремонт, bitflip", "ga_repair_swap": "ГА: ремонт, swap",
          "ga_repair_nofill": "ГА: ремонт без дозаполнения", "ga_penalty": "ГА: штраф λ=300",
          "ga_penalty_swap": "ГА: штраф λ=300, swap",
          "ga_penalty_low": "ГА: штраф λ=50", "random_feasible": "случайный допустимый поиск",
          "greedy": "жадная эвристика"}
TESTS = [("ga_repair", "ga_penalty"), ("ga_penalty", "ga_penalty_low"), ("ga_penalty", "ga_penalty_swap"),
         ("ga_repair_swap", "ga_repair"),
         ("ga_repair", "ga_repair_nofill"), ("ga_repair", "random_feasible")]


def load_summary(tag):
    rows = read_csv(RESULTS / f"{tag}_summary.csv")
    s = {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
         for k in ("best_value", "feasible", "evals", "time_s")}
    return s


def load_curves(tag, ycol="best_so_far"):
    by_run, xs = defaultdict(list), defaultdict(list)
    for r in read_csv(RESULTS / f"{tag}_gens.csv"):
        by_run[r["run"]].append(float(r[ycol]) if r[ycol] != "" else np.nan)
        xs[r["run"]].append(float(r["evals"]))
    return np.array(next(iter(xs.values()))), np.array(list(by_run.values()))


def evals_to_reach(tag, level):
    """Медиана по запускам числа вызовов ФФ до первого best-so-far ≥ level (∞, если не достигнут)."""
    x, M = load_curves(tag)
    first = [x[np.argmax(row >= level)] if np.any(row >= level) else np.inf for row in np.nan_to_num(M, nan=-1)]
    return np.median(first)


def table(tags, opt):
    lines = ["| метод | best | mean | median | std | worst | отклонение медианы от опт., % | "
             "найден оптимум | допустимо | вызовов до 99 % опт. (медиана) | время/запуск, с | вызовов ФФ |",
             "|" + "---|" * 12]
    for t in tags:
        s = load_summary(t)
        v = s["best_value"][s["feasible"] == 1]
        std = v.std(ddof=1) if v.size > 1 else 0.0
        lines.append(
            f"| {LABELS[t]} | {v.max():.1f} | {v.mean():.1f} | {np.median(v):.1f} | {std:.2f} | {v.min():.1f} | "
            f"{100 * (opt - np.median(v)) / opt:.2f} | {int(np.isclose(v, opt).sum())}/{s['best_value'].size} | "
            f"{100 * s['feasible'].mean():.0f}% | {evals_to_reach(t, 0.99 * opt):.0f} | "
            f"{s['time_s'].mean():.3f} | {int(s['evals'].mean())} |")
    return lines


def tests(tags):
    lines = ["| A vs B | медиана A | медиана B | U | z | p |", "|---|---|---|---|---|---|"]
    for a, b in TESTS:
        if a in tags and b in tags:
            fa, fb = load_summary(a)["best_value"], load_summary(b)["best_value"]
            u, z, p = mann_whitney(fa, fb)  # z > 0 — у A значения больше (лучше при максимизации)
            lines.append(f"| {LABELS[a]} vs {LABELS[b]} | {np.nanmedian(fa):.1f} | {np.nanmedian(fb):.1f} | "
                         f"{u:.0f} | {z:.2f} | {p:.2e} |")
    return lines


def plots(tags, opt):
    ref = (opt, f"точный оптимум {opt:.1f}")
    kw = dict(log=False, fmt="{:.1f}", ref=ref, ylabel="лучшая допустимая полезность")
    groups = [("cmp_constraints.png", ["ga_repair", "ga_penalty", "ga_penalty_low"],
               "Учёт ограничений: ремонт vs штраф (два λ)", (330, opt + 8)),
              ("cmp_mutation.png", ["ga_penalty", "ga_penalty_swap"],
               "Мутация при штрафе λ=300: bitflip vs swap", (440, opt + 5)),
              ("cmp_mutation_repair.png", ["ga_repair", "ga_repair_swap"],
               "Мутация при ремонте: bitflip vs swap", (495, opt + 3)),
              ("cmp_baseline.png", ["ga_repair", "ga_penalty", "random_feasible"],
               "ГА vs случайный допустимый поиск (20 000 вызовов)", (470, opt + 5))]
    for fname, group, title, ylim in groups:
        if all(t in tags for t in group):
            plotting.compare([(LABELS[t], *load_curves(t)) for t in group], title, RESULTS / fname,
                             ylim=ylim, **kw)
    pen = [t for t in ("ga_penalty", "ga_penalty_low") if t in tags]
    if pen:
        plotting.compare([(LABELS[t], *load_curves(t, "feasible_frac")) for t in pen],
                         "Доля допустимых особей в популяции (штрафные ГА)", RESULTS / "feasible_frac.png",
                         log=False, fmt="{:.2f}", ylabel="доля допустимых", ylim=(0, 1.05))
    plotting.final_boxes([(LABELS[t].replace("ГА: ", "").replace(" ", "\n", 1), load_summary(t)["best_value"])
                          for t in tags if t != "greedy"],
                         "Итоговая полезность 20 запусков", RESULTS / "final_boxes.png",
                         ylabel="лучшая допустимая полезность", log=False, ref=ref)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()
    opt = json.loads((RESULTS / "exact.json").read_text())["value"]
    tags = [t for t in ORDER if (RESULTS / f"{t}_summary.csv").exists()]
    t1, t2 = table(tags, opt), tests(tags)
    (RESULTS / "summary.md").write_text("\n".join([f"Точный оптимум (ветви и границы): **{opt:.1f}**", "",
                                                   "## Сводка серий", "", *t1, "",
                                                   "## U-тест Манна—Уитни (z > 0: A лучше)", "", *t2, ""]))
    for line in t1[2:]:
        c = [x.strip() for x in line.split("|")[1:-1]]
        print(f"{c[0]:<30} med={c[3]:>6} gap={c[6]:>5}% opt={c[7]:>5} feas={c[8]} to99%={c[9]}")
    if args.plots:
        plots(tags, opt)


if __name__ == "__main__":
    main()

"""ЛР2: примеры допустимых/недопустимых решений и лучший комплект в предметной форме.

Пример: python -m lab2.examples  -> results/examples.md
"""
import json
from pathlib import Path

import numpy as np

from evo_core.stats import read_csv
from lab2.operators import Repairer
from lab2.problem import load_instance, totals, violations

RESULTS = Path(__file__).parent / "results"


def describe(inst, x, title):
    t, v = totals(inst, x), {k: float(a[0]) for k, a in violations(inst, x).items()}
    sel = np.flatnonzero(x)
    bad_conf = [f"{inst.names[a]} + {inst.names[b]}" for a, b in inst.conf if x[a] and x[b]]
    bad_req = [f"{inst.names[a]} без «{inst.names[b]}»" for a, b in inst.req if x[a] and not x[b]]
    ok = sum(v.values()) == 0
    lines = [f"### {title} — {'допустимо' if ok else 'НЕДОПУСТИМО'}", "",
             f"Строка: `{''.join(map(str, x))}` ({t['items']} приборов, id: {', '.join(map(str, sel))})", "",
             f"- полезность {t['value']:.1f}; стоимость {t['cost']:.0f} / {inst.budget:.0f} тыс. ₽; "
             f"мощность {t['power']:.2f} / {inst.power_limit:.2f} кВт"]
    if v["budget"] > 0:
        lines.append(f"- превышен бюджет на {100 * v['budget']:.1f} %")
    if v["power"] > 0:
        lines.append(f"- превышена мощность на {100 * v['power']:.1f} %")
    if bad_conf:
        lines.append(f"- несовместимы: {'; '.join(bad_conf)}")
    if bad_req:
        lines.append(f"- нарушены зависимости: {'; '.join(bad_req)}")
    return lines + [""]


def kit_table(inst, x):
    lines = ["| id | прибор | категория | стоимость, тыс. ₽ | мощность, кВт | полезность |",
             "|---|---|---|---|---|---|"]
    for i in sorted(np.flatnonzero(x), key=lambda i: (inst.categories[i], -inst.value[i])):
        lines.append(f"| {i} | {inst.names[i]} | {inst.categories[i]} | {inst.cost[i]:.0f} | "
                     f"{inst.power[i]:.2f} | {inst.value[i]:.1f} |")
    t = totals(inst, x)
    lines.append(f"| | **итого** | | **{t['cost']:.0f}** из {inst.budget:.0f} | "
                 f"**{t['power']:.2f}** из {inst.power_limit:.2f} | **{t['value']:.1f}** |")
    return lines


def main():
    inst = load_instance()
    rep = Repairer(inst)
    rng = np.random.default_rng(7)
    exact = np.array(json.loads((RESULTS / "exact.json").read_text())["x"], dtype=np.int8)
    rows = [dict(r, tag=f.name.removesuffix("_summary.csv"))
            for f in sorted(RESULTS.glob("ga_*_summary.csv")) for r in read_csv(f) if r["x"]]
    best = max(rows, key=lambda r: float(r["best_value"]))  # первое по алфавиту среди равных
    ga_best = np.array(list(map(int, best["x"])), dtype=np.int8)
    greedy = rep.repair_one(np.zeros(inst.n, dtype=np.int8))

    rand = (rng.random(inst.n) < 0.5).astype(np.int8)          # случайная строка — типичная особь без ремонта
    broken = ga_best.copy()                                      # лучший комплект + конфликтующий прибор
    broken[2] = 1                                                # конфокальный: конфликт с флуоресц. и ИК, нужна станция
    out = ["# ЛР2: примеры решений", "",
           "Приборы, требования и несовместимости — `data/instance.json`.", "",
           "## Допустимые решения", "",
           *describe(inst, greedy, "Жадная эвристика по удельной полезности"),
           *describe(inst, ga_best, f"Лучшее решение ГА ({best['tag']}, seed {best['seed']})"),
           "## Недопустимые решения", "",
           *describe(inst, rand, "Случайная строка (p = 0.5)"),
           *describe(inst, rep.repair_one(rand), "…та же строка после ремонта"),
           *describe(inst, broken, "Лучший комплект + конфокальный микроскоп"),
           "## Лучший найденный комплект", "",
           f"Совпадает с точным оптимумом: **{'да' if np.array_equal(ga_best, exact) else 'нет'}**.", "",
           *kit_table(inst, ga_best), ""]
    (RESULTS / "examples.md").write_text("\n".join(out))
    print(f"examples.md: ГА-лучшее {totals(inst, ga_best)['value']:.1f}, совпадает с оптимумом: "
          f"{np.array_equal(ga_best, exact)}")


if __name__ == "__main__":
    main()

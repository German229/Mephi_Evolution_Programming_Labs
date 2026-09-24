"""ЛР2: серия независимых запусков по одному или нескольким конфигам.

Пример: python -m lab2.main --config lab2/configs/ga_repair.toml --runs 20 --seed 2000
"""
from pathlib import Path

import numpy as np

from evo_core.cli import load_config, parse_args, run_seeds, save_effective_config
from evo_core.stats import fmt, write_csv
from lab2.ga import ALGORITHMS, LOG_COLUMNS
from lab2.problem import is_feasible, load_instance, totals

RESULTS = Path(__file__).parent / "results"


def run_series(cfg, out):
    inst = load_instance(cfg["instance"])
    algo = ALGORITHMS[cfg["algorithm"]]
    tag = cfg["tag"]
    gen_rows, sum_rows = [], []
    for run, seed in enumerate(run_seeds(cfg)):
        r = algo(cfg, seed, inst)
        gen_rows += [(run, seed, *map(fmt, row)) for row in r["log"]]
        x = r["best_x"]
        ok = x is not None and bool(is_feasible(inst, x)[0])  # независимая перепроверка допустимости
        t = totals(inst, x) if x is not None else {"cost": np.nan, "power": np.nan, "items": 0}
        sum_rows.append((run, seed, fmt(r["best_value"]), int(ok), r["evals"], r["gens"], fmt(r["time_s"]),
                         fmt(t["cost"]), fmt(t["power"]), t["items"],
                         "" if x is None else "".join(map(str, x))))
    write_csv(out / f"{tag}_gens.csv", ["run", "seed", *LOG_COLUMNS], gen_rows)
    write_csv(out / f"{tag}_summary.csv", ["run", "seed", "best_value", "feasible", "evals", "gens",
                                           "time_s", "cost", "power", "items", "x"], sum_rows)
    save_effective_config(cfg, out / f"{tag}_config.json")
    v = np.array([float(r[2]) for r in sum_rows])
    print(f"{tag:<17} runs={len(v)} median={np.nanmedian(v):.1f} best={np.nanmax(v):.1f} "
          f"worst={np.nanmin(v):.1f} feasible={sum(r[3] for r in sum_rows)}/{len(v)} "
          f"t={sum(float(r[6]) for r in sum_rows):.1f}s")


def main():
    args = parse_args("ЛР2: ГА для выбора комплекта оборудования")
    out = Path(args.out) if args.out else RESULTS
    for path in args.config:
        run_series(load_config(path, args.seed, args.runs), out)


if __name__ == "__main__":
    main()

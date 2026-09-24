"""ЛР1: серия независимых запусков по одному или нескольким конфигам.

Пример: python -m lab1.main --config lab1/configs/ga_base.toml --runs 20 --seed 1000
"""
from pathlib import Path

import numpy as np

from evo_core.cli import load_config, parse_args, run_seeds, save_effective_config
from evo_core.stats import fmt, write_csv
from lab1.ga import ALGORITHMS, LOG_COLUMNS

RESULTS = Path(__file__).parent / "results"


def run_series(cfg, out):
    algo = ALGORITHMS[cfg["algorithm"]]
    tag = cfg["tag"]
    gen_rows, sum_rows = [], []
    for run, seed in enumerate(run_seeds(cfg)):
        r = algo(cfg, seed)
        gen_rows += [(run, seed, *map(fmt, row)) for row in r["log"]]
        sum_rows.append((run, seed, fmt(r["best_f"]), r["evals"], r["gens"], fmt(r["time_s"]),
                         int(r["feasible"]), ";".join(f"{v:.6g}" for v in r["best_x"])))
    write_csv(out / f"{tag}_gens.csv", ["run", "seed", *LOG_COLUMNS], gen_rows)
    write_csv(out / f"{tag}_summary.csv",
              ["run", "seed", "best_f", "evals", "gens", "time_s", "feasible", "best_x"], sum_rows)
    save_effective_config(cfg, out / f"{tag}_config.json")
    best = np.array([float(r[2]) for r in sum_rows])
    total_t = sum(float(r[5]) for r in sum_rows)
    print(f"{tag:<16} runs={len(best)} median={np.median(best):.3e} best={best.min():.3e} "
          f"worst={best.max():.3e} t={total_t:.1f}s")


def main():
    args = parse_args("ЛР1: ГА для функции Гриванка")
    out = Path(args.out) if args.out else RESULTS
    for path in args.config:
        run_series(load_config(path, args.seed, args.runs), out)


if __name__ == "__main__":
    main()

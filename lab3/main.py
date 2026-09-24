"""ЛР3: серии независимых запусков (NSGA-II, взвешенная свёртка, случайный поиск).

Пример: python -m lab3.main --config lab3/configs/nsga2.toml --runs 20 --seed 3000
"""
from pathlib import Path

import numpy as np

from evo_core.cli import load_config, parse_args, run_seeds, save_effective_config
from evo_core.stats import fmt, write_csv
from lab3.nsga2 import ALGORITHMS, LOG_COLUMNS

RESULTS = Path(__file__).parent / "results"


def run_series(cfg, out):
    algo = ALGORITHMS[cfg["algorithm"]]
    tag = cfg["tag"]
    gen_rows, sum_rows, front_rows, res = [], [], [], []
    for run, seed in enumerate(run_seeds(cfg)):
        r = algo(cfg, seed)
        res.append(r)
        gen_rows += [(run, seed, *map(fmt, row)) for row in r["log"]]
        front_rows += [(run, seed, *map(fmt, f)) for f in r["F"]]
        sum_rows.append((run, seed, fmt(r["hv"]), r["front_size"], r["evals"], r["gens"], fmt(r["time_s"]),
                         *map(fmt, r["F"].min(0))))
    # решения (расписания) сохраняются только для запуска с медианным гиперобъёмом — представительный, не лучший
    med = int(np.argsort([r["hv"] for r in res])[len(res) // 2])
    x_rows = [(i, *map(fmt, f), *map(fmt, x)) for i, (f, x) in enumerate(zip(res[med]["F"], res[med]["X"]))]
    days = res[med]["X"].shape[1]
    write_csv(out / f"{tag}_gens.csv", ["run", "seed", *LOG_COLUMNS], gen_rows)
    write_csv(out / f"{tag}_summary.csv", ["run", "seed", "hv", "front_size", "evals", "gens", "time_s",
                                           "min_water", "min_deficit", "min_wet"], sum_rows)
    write_csv(out / f"{tag}_fronts.csv", ["run", "seed", "water", "deficit", "wet"], front_rows)
    write_csv(out / f"{tag}_median_front.csv",
              ["i", "water", "deficit", "wet", *[f"I{t + 1}" for t in range(days)]], x_rows)
    save_effective_config(dict(cfg, median_run=med), out / f"{tag}_config.json")
    hv = np.array([r["hv"] for r in res])
    print(f"{tag:<16} runs={len(hv)} HV median={np.median(hv):.4f} best={hv.max():.4f} worst={hv.min():.4f} "
          f"front={np.median([r['front_size'] for r in res]):.0f} t={sum(r['time_s'] for r in res):.1f}s")


def main():
    args = parse_args("ЛР3: NSGA-II для режима полива")
    out = Path(args.out) if args.out else RESULTS
    for path in args.config:
        run_series(load_config(path, args.seed, args.runs), out)


if __name__ == "__main__":
    main()

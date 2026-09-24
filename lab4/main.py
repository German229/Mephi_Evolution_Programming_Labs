"""ЛР4: серии независимых запусков ГП по всем задачам конфига.

Пример: python -m lab4.main --config lab4/configs/gp_nsga2.toml --runs 30 --seed 4000
"""
from pathlib import Path

import numpy as np

from evo_core.cli import load_config, parse_args, run_seeds, save_effective_config
from evo_core.stats import fmt, write_csv
from lab4.gp import LOG_COLUMNS, run_gp

RESULTS = Path(__file__).parent / "results"
SUMMARY = ["run", "seed", "train", "val", "test", "extrap", "size", "depth", "evals", "gens", "nodes", "time_s",
           "a", "b", "expr"]


def run_series(cfg, problem, out):
    tag = f"{problem}_{cfg['tag']}"
    gen_rows, sum_rows, front_rows = [], [], []
    for run, seed in enumerate(run_seeds(cfg)):
        r = run_gp(cfg, seed, problem)
        gen_rows += [(run, seed, *map(fmt, row)) for row in r["log"]]
        sum_rows.append((run, seed, *[fmt(r[k]) for k in SUMMARY[2:-1]], r["expr"]))
        front_rows += [(run, seed, s, fmt(e), fmt(v), expr) for s, e, v, expr in r.get("front", [])]
    write_csv(out / f"{tag}_gens.csv", ["run", "seed", *LOG_COLUMNS], gen_rows)
    write_csv(out / f"{tag}_summary.csv", SUMMARY, sum_rows)
    if front_rows:
        write_csv(out / f"{tag}_fronts.csv", ["run", "seed", "size", "train", "val", "expr"], front_rows)
    save_effective_config(dict(cfg, problem=problem), out / f"{tag}_config.json")
    te = np.array([float(r[4]) for r in sum_rows])
    sz = np.array([float(r[6]) for r in sum_rows])
    print(f"{tag:<22} runs={len(te)} test NRMSE med={np.median(te):.4f} size med={np.median(sz):.0f} "
          f"t={sum(float(r[11]) for r in sum_rows):.0f}s")


def main():
    args = parse_args("ЛР4: ГП для символьной регрессии с контролем bloat")
    out = Path(args.out) if args.out else RESULTS
    for path in args.config:
        cfg = load_config(path, args.seed, args.runs)
        for problem in cfg["problems"]:
            run_series(cfg, problem, out)


if __name__ == "__main__":
    main()

"""ЛР1: предварительная развёртка σ на seed 0..9 (отдельно от итоговых seed 1000+).

Пример: python -m lab1.sweep   (~2 с) -> results/sweep_sigma.csv
"""
import copy
from pathlib import Path

import numpy as np

from evo_core.cli import load_config
from evo_core.stats import fmt, write_csv
from lab1.ga import run_ga

SIGMAS = [0.1, 0.03, 0.01, 0.003, 0.001]
SEEDS = range(10)


def main():
    base = load_config(Path(__file__).parent / "configs" / "ga_base.toml")
    rows = []
    for adaptive in (False, True):
        for s in SIGMAS:
            cfg = copy.deepcopy(base)
            cfg["ga"].update(sigma_frac=s, adaptive=adaptive)
            v = np.array([run_ga(cfg, seed)["best_f"] for seed in SEEDS])
            rows.append((int(adaptive), s, fmt(np.median(v)), fmt(v.mean()), fmt(v.min()), int((v < 1e-8).sum())))
            print(f"adaptive={int(adaptive)} sigma0={s:<6} median={np.median(v):.2e} hits={rows[-1][-1]}/10")
    write_csv(Path(__file__).parent / "results" / "sweep_sigma.csv",
              ["adaptive", "sigma_frac", "median", "mean", "best", "hits_1e-8"], rows)


if __name__ == "__main__":
    main()

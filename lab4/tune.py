"""ЛР4: подбор параметров контроля bloat (c для парсимонии, p для тарпейского метода)
на seed 100..102, не пересекающихся с итоговыми сериями (4000+). Выбор — компромисс медиан validation NRMSE
и размера модели по обеим задачам (строгий минимум validation NRMSE давал бы в 3–9 раз более крупные деревья).

Пример: python -m lab4.tune  (~1.5 мин) -> results/tuning.csv
"""
import copy
from pathlib import Path

import numpy as np

from evo_core.cli import load_config
from evo_core.stats import fmt, write_csv
from lab4.gp import run_gp

GRID = [("parsimony", "parsimony_c", [3e-4, 1e-3, 3e-3]), ("tarpeian", "tarpeian_p", [0.3, 0.6, 0.9])]
SEEDS = [100, 101, 102]


def main():
    base = load_config(Path(__file__).parent / "configs" / "gp_depth.toml")
    rows = []
    for mode, key, values in GRID:
        for v in values:
            for prob in base["problems"]:
                cfg = copy.deepcopy(base)
                cfg["bloat_control"] = mode
                cfg["gp"][key] = v
                rs = [run_gp(cfg, s, prob) for s in SEEDS]
                val = np.median([r["val"] for r in rs])
                size = np.median([r["size"] for r in rs])
                rows.append((mode, key, v, prob, fmt(val), fmt(size)))
                print(f"{mode:<9} {key}={v:<7} {prob:<8} val={val:.4f} size={size:.0f}")
    write_csv(Path(__file__).parent / "results" / "tuning.csv",
              ["mode", "param", "value", "problem", "val_nrmse_median", "size_median"], rows)


if __name__ == "__main__":
    main()

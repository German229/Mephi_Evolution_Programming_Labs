"""ЛР3: синтетический прогноз погоды: эталонная эвапотранспирация ET0 (известна) и ансамбль
из K сценариев осадков (неизвестны заранее — расписание должно быть устойчиво ко всем).

Пример: python -m lab3.generate_data --seed 2026 -> lab3/data/weather.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

DAYS = 30
SCENARIOS = 10


def generate(seed, days=DAYS, k=SCENARIOS):
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    # ET0: жаркая середина периода + дневной шум, мм/сут
    et0 = np.clip(5.0 + 1.5 * np.sin(np.pi * t / (days - 1)) + rng.normal(0, 0.6, days), 2.0, 8.5)
    # осадки, K сценариев: редкие дожди (p = 0.2, гамма) + с вероятностью 0.5 ливень 45 мм в случайный день
    rain = np.where(rng.random((k, days)) < 0.2, rng.gamma(0.8, 15.0, (k, days)), 0.0)
    storm = rng.random(k) < 0.5
    rain[storm, rng.integers(0, days, storm.sum())] += 45.0
    return {"seed": seed, "days": days, "scenarios": k, "units": "мм/сут",
            "et0": np.round(et0, 2).tolist(), "rain": np.round(rain, 1).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default=str(Path(__file__).parent / "data" / "weather.json"))
    args = p.parse_args()
    w = generate(args.seed)
    Path(args.out).write_text(json.dumps(w, ensure_ascii=False, indent=1))
    r = np.array(w["rain"])
    print(f"days={w['days']} K={w['scenarios']} ΣET0={sum(w['et0']):.0f} мм "
          f"Σrain по сценариям: {r.sum(1).min():.0f}..{r.sum(1).max():.0f} мм (среднее {r.sum(1).mean():.0f})")


if __name__ == "__main__":
    main()

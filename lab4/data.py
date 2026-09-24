"""ЛР4: эталонные задачи символьной регрессии и их разбиение train/val/test + экстраполяция.

Пример: python -m lab4.data --seed 2026  -> lab4/data/<задача>.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
NOISE = 0.05  # σ шума = 5 % стандартного отклонения чистого сигнала


def nguyen7(X):
    return np.log(X[:, 0] + 1) + np.log(X[:, 0] ** 2 + 1)


def pagie1(X):
    with np.errstate(divide="ignore"):
        return 1 / (1 + X[:, 0] ** -4.0) + 1 / (1 + X[:, 1] ** -4.0)


PROBLEMS = {
    # имя: (функция, число переменных, область обучения, область экстраполяции, n точек, n экстраполяции)
    "nguyen7": (nguyen7, 1, (0.0, 2.0), (2.0, 3.0), 200, 50),
    "pagie1": (pagie1, 2, (-5.0, 5.0), (5.0, 7.0), 400, 100),
}
TRUE_FORMULA = {"nguyen7": "ln(x0 + 1) + ln(x0^2 + 1)", "pagie1": "1/(1 + x0^-4) + 1/(1 + x1^-4)"}


def generate(name, seed):
    f, d, (lo, hi), (elo, ehi), n, n_ext = PROBLEMS[name]
    rng = np.random.default_rng(seed)
    X = rng.uniform(lo, hi, (n, d))
    y_clean = f(X)
    y = y_clean + rng.normal(0, NOISE * y_clean.std(), n)
    split = np.array(["train"] * int(0.6 * n) + ["val"] * int(0.2 * n) + ["test"] * (n - int(0.8 * n)))
    rng.shuffle(split)
    # экстраполяция: точки вне обучающего куба (|x| за пределами по случайной координате), без шума
    Xe = rng.uniform(lo, hi, (n_ext, d))
    k = rng.integers(0, d, n_ext)
    side = rng.random(n_ext) < 0.5 if lo < 0 else np.ones(n_ext, bool)
    mag = rng.uniform(elo, ehi, n_ext)
    Xe[np.arange(n_ext), k] = np.where(side, mag, -mag) if lo < 0 else mag
    ye = f(Xe)
    rows = [(*x, yv, s) for x, yv, s in zip(X, y, split)] + [(*x, yv, "extrap") for x, yv in zip(Xe, ye)]
    return d, rows


def save(name, seed):
    d, rows = generate(name, seed)
    DATA.mkdir(exist_ok=True)
    with (DATA / f"{name}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([*[f"x{i}" for i in range(d)], "y", "split"])
        w.writerows([(*[f"{v:.6g}" for v in r[:-1]], r[-1]) for r in rows])
    return len(rows)


def load(name):
    """Возвращает X (n, d), y (n,), словарь масок по разбиениям."""
    with (DATA / f"{name}.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    xs = [k for k in rows[0] if k.startswith("x")]
    X = np.array([[float(r[k]) for k in xs] for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    split = np.array([r["split"] for r in rows])
    return X, y, {s: split == s for s in ("train", "val", "test", "extrap")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    for name in PROBLEMS:
        print(f"{name}: {save(name, args.seed)} точек -> data/{name}.csv")


if __name__ == "__main__":
    main()

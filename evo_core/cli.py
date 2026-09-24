"""Общий CLI: конфиги TOML, seed, число запусков."""
import argparse
import json
import tomllib
from pathlib import Path


def parse_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", nargs="+", required=True, help="один или несколько .toml")
    p.add_argument("--seed", type=int, help="базовый seed; запуск i использует seed + i")
    p.add_argument("--runs", type=int, help="число независимых запусков")
    p.add_argument("--out", default=None, help="каталог результатов")
    return p.parse_args()


def load_config(path, seed=None, runs=None):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if seed is not None:
        cfg["seed"] = seed
    if runs is not None:
        cfg["runs"] = runs
    return cfg


def run_seeds(cfg):
    return [cfg["seed"] + i for i in range(cfg["runs"])]


def save_effective_config(cfg, path):
    """Фактически использованные параметры (с учётом CLI) — для воспроизводимости."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

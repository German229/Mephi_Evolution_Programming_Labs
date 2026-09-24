"""ЛР2: модель задачи выбора комплекта оборудования.

Решение — бинарная строка x ∈ {0,1}^n (x_i = 1 — прибор i закупается).
max V(x) = Σ v_i x_i  при  Σ c_i x_i ≤ B,  Σ w_i x_i ≤ P,
x_a + x_b ≤ 1 для несовместимых (a, b),  x_a ≤ x_b для «a требует b».
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_INSTANCE = Path(__file__).parent / "data" / "instance.json"


@dataclass
class Instance:
    names: list
    categories: list
    cost: np.ndarray
    power: np.ndarray
    value: np.ndarray
    budget: float
    power_limit: float
    req: np.ndarray        # (k, 2): [a, b] — a требует b
    conf: np.ndarray       # (m, 2): [a, b] — нельзя вместе
    raw: dict

    @property
    def n(self):
        return len(self.value)

    @property
    def ratio(self):
        """Удельная полезность на нормированный расход обоих ресурсов."""
        return self.value / (self.cost / self.budget + self.power / self.power_limit)


def load_instance(path=DEFAULT_INSTANCE):
    d = json.loads(Path(path).read_text())
    it = d["items"]
    return Instance(
        names=[x["name"] for x in it], categories=[x["category"] for x in it],
        cost=np.array([x["cost"] for x in it]), power=np.array([x["power"] for x in it]),
        value=np.array([x["value"] for x in it]), budget=d["budget"], power_limit=d["power_limit"],
        req=np.array([[r["item"], r["needs"]] for r in d["requires"]], dtype=int).reshape(-1, 2),
        conf=np.array([[c["a"], c["b"]] for c in d["conflicts"]], dtype=int).reshape(-1, 2),
        raw=d,
    )


def violations(inst, X):
    """Покомпонентные нарушения для строк X (pop, n).

    Возвращает dict массивов: относительные превышения бюджета и мощности,
    число нарушенных несовместимостей и зависимостей.
    """
    X = np.atleast_2d(X)
    cost, power = X @ inst.cost, X @ inst.power
    return {
        "budget": np.maximum(0.0, cost - inst.budget) / inst.budget,
        "power": np.maximum(0.0, power - inst.power_limit) / inst.power_limit,
        "conflicts": (X[:, inst.conf[:, 0]] & X[:, inst.conf[:, 1]]).sum(1),
        "requires": (X[:, inst.req[:, 0]] & (1 - X[:, inst.req[:, 1]])).sum(1),
    }


def total_violation(v):
    return v["budget"] + v["power"] + v["conflicts"] + v["requires"]


def is_feasible(inst, X):
    return total_violation(violations(inst, X)) == 0


def totals(inst, x):
    x = np.asarray(x)
    return {"value": float(x @ inst.value), "cost": float(x @ inst.cost), "power": float(x @ inst.power),
            "items": int(x.sum())}

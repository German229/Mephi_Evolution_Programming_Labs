"""ЛР2: точный оптимум собственным методом ветвей и границ (для оценки качества ГА).

Граница — минимум из двух дробных (Данцига) релаксаций: по бюджету и по мощности,
по ещё не решённым приборам, совместимым с уже выбранными.
"""
import time

import numpy as np

from lab2.operators import Repairer


def _frac_bound(items, v, w, cap):
    """Дробный рюкзак по одному ресурсу; items уже отсортированы по v/w убыв."""
    total = 0.0
    for i in items:
        if w[i] <= cap:
            cap -= w[i]
            total += v[i]
        else:
            return total + v[i] * cap / w[i]
    return total


def solve_exact(inst, time_limit=120.0):
    rep = Repairer(inst, fill=True)
    n = inst.n
    v, c, p = inst.value.tolist(), inst.cost.tolist(), inst.power.tolist()
    order = list(np.argsort(-inst.ratio))                       # порядок ветвления
    by_c = sorted(range(n), key=lambda i: -v[i] / c[i])
    by_p = sorted(range(n), key=lambda i: -v[i] / p[i])
    pos = {i: k for k, i in enumerate(order)}

    greedy = rep.repair_one(np.zeros(n, dtype=np.int8))          # стартовый рекорд
    best = [float(greedy @ inst.value), set(np.flatnonzero(greedy).tolist())]
    state = {"nodes": 0, "t0": time.perf_counter(), "timeout": False}
    decided = {}                                                  # i -> 0/1

    def consistent(i, val):
        if val == 1:
            if any(decided.get(cf) == 1 for cf in rep.conf_with[i]):
                return False
            if any(decided.get(b) == 0 for b in rep.needs[i]):
                return False
        else:
            if any(decided.get(a) == 1 for a in rep.needed_by[i]):
                return False
        return True

    def bound(k, cost_left, power_left, val):
        free = set(order[k:])
        free = {i for i in free if not any(decided.get(cf) == 1 for cf in rep.conf_with[i])}
        b1 = _frac_bound([i for i in by_c if i in free], v, c, cost_left)
        b2 = _frac_bound([i for i in by_p if i in free], v, p, power_left)
        return val + min(b1, b2)

    def dfs(k, cost_left, power_left, val):
        state["nodes"] += 1
        if state["nodes"] % 2048 == 0 and time.perf_counter() - state["t0"] > time_limit:
            state["timeout"] = True
        if state["timeout"]:
            return
        if k == n:
            if val > best[0]:
                best[0], best[1] = val, {i for i, d in decided.items() if d == 1}
            return
        if bound(k, cost_left, power_left, val) <= best[0] + 1e-9:
            return
        i = order[k]
        if c[i] <= cost_left and p[i] <= power_left and consistent(i, 1):
            decided[i] = 1
            dfs(k + 1, cost_left - c[i], power_left - p[i], val + v[i])
            del decided[i]
        if consistent(i, 0):
            decided[i] = 0
            dfs(k + 1, cost_left, power_left, val)
            del decided[i]

    dfs(0, inst.budget, inst.power_limit, 0.0)
    x = np.zeros(n, dtype=np.int8)
    x[list(best[1])] = 1
    return {"value": best[0], "x": x, "nodes": state["nodes"], "optimal": not state["timeout"],
            "time_s": time.perf_counter() - state["t0"], "pos": pos}


def main():
    """python -m lab2.exact -> results/exact.json (оптимум для расчёта отклонения ГА)."""
    import json
    from pathlib import Path

    from lab2.problem import is_feasible, load_instance, totals

    inst = load_instance()
    r = solve_exact(inst)
    out = Path(__file__).parent / "results" / "exact.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"value": r["value"], "optimal": r["optimal"], "nodes": r["nodes"],
                               "time_s": round(r["time_s"], 3), "x": r["x"].tolist(),
                               **{k: v for k, v in totals(inst, r["x"]).items() if k != "value"}},
                              ensure_ascii=False))
    print(f"exact value={r['value']:.1f} optimal={r['optimal']} nodes={r['nodes']} "
          f"t={r['time_s']:.2f}s feasible={bool(is_feasible(inst, r['x'])[0])}")


if __name__ == "__main__":
    main()

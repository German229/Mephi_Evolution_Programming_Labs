"""ЛР2: операторы для бинарного представления и обработка ограничений."""
import numpy as np

from lab2.problem import total_violation, violations


# ---------- кроссовер ----------
def uniform_crossover(rng, P1, P2):
    mask = rng.random(P1.shape) < 0.5
    return np.where(mask, P1, P2), np.where(mask, P2, P1)


def two_point_crossover(rng, P1, P2):
    n = P1.shape[1]
    a, b = np.sort(rng.integers(0, n + 1, size=(2, len(P1))), axis=0)
    pos = np.arange(n)
    mask = (pos >= a[:, None]) & (pos < b[:, None])
    return np.where(mask, P2, P1), np.where(mask, P1, P2)


# ---------- мутация ----------
def bitflip_mutation(rng, X, p_gene):
    """Каждый бит инвертируется с вероятностью p_gene (обычно 1/n)."""
    flip = (rng.random(X.shape) < p_gene).astype(np.int8)
    return X ^ flip


def swap_mutation(rng, X, p_ind):
    """Обмен «взять/убрать»: один выбранный прибор заменяется одним невыбранным.

    Сохраняет число приборов и примерно сохраняет загрузку ресурсов — ход по границе
    допустимой области, которого битовая мутация делает только за два удачных шага.
    """
    X = X.copy()
    for k in np.nonzero(rng.random(len(X)) < p_ind)[0]:
        ones, zeros = np.flatnonzero(X[k]), np.flatnonzero(X[k] == 0)
        if ones.size and zeros.size:
            X[k, rng.choice(ones)] = 0
            X[k, rng.choice(zeros)] = 1
        elif zeros.size:
            X[k, rng.choice(zeros)] = 1
    return X


# ---------- ремонт ----------
class Repairer:
    """Жадный детерминированный ремонт + (опционально) жадное дозаполнение.

    1) замыкание по зависимостям: прибор без требуемого — убирается;
    2) несовместимые пары: убирается прибор с меньшей удельной полезностью;
    3) пока превышен бюджет или мощность — убирается прибор с худшей удельной полезностью
       (после каждого удаления снова шаг 1);
    4) дозаполнение: невыбранные приборы в порядке убывания удельной полезности
       добавляются, если не нарушают ни одного ограничения.
    Результат всегда допустим; ремонт ламарковский — исправленная строка заменяет генотип.
    """

    def __init__(self, inst, fill=True):
        self.inst, self.fill = inst, fill
        self.order_worst = list(np.argsort(inst.ratio))          # от худшего к лучшему
        self.order_best = self.order_worst[::-1]
        self.needs = {i: [] for i in range(inst.n)}              # i требует needs[i]
        self.needed_by = {i: [] for i in range(inst.n)}
        for a, b in inst.req:
            self.needs[a].append(b)
            self.needed_by[b].append(a)
        self.conf_with = {i: [] for i in range(inst.n)}
        for a, b in inst.conf:
            self.conf_with[a].append(b)
            self.conf_with[b].append(a)
        self.cost, self.power = inst.cost.tolist(), inst.power.tolist()
        self.ratio = inst.ratio.tolist()

    def _drop(self, sel, i):
        """Удалить i и всё, что (транзитивно) от него зависит."""
        stack = [i]
        while stack:
            j = stack.pop()
            if j in sel:
                sel.discard(j)
                stack.extend(self.needed_by[j])

    def repair_one(self, x):
        sel = set(np.flatnonzero(x).tolist())
        for i in list(sel):                                        # 1) зависимости
            if i in sel and any(b not in sel for b in self.needs[i]):
                self._drop(sel, i)
        for a, b in self.inst.conf:                                # 2) несовместимости
            if a in sel and b in sel:
                self._drop(sel, a if self.ratio[a] < self.ratio[b] else b)
        cost = sum(self.cost[i] for i in sel)
        power = sum(self.power[i] for i in sel)
        for i in self.order_worst:                                 # 3) ресурсы
            if cost <= self.inst.budget and power <= self.inst.power_limit:
                break
            if i in sel:
                before = set(sel)
                self._drop(sel, i)
                for j in before - sel:
                    cost -= self.cost[j]
                    power -= self.power[j]
        if self.fill:                                              # 4) дозаполнение
            for i in self.order_best:
                if (i not in sel and cost + self.cost[i] <= self.inst.budget
                        and power + self.power[i] <= self.inst.power_limit
                        and all(b in sel for b in self.needs[i])
                        and not any(c in sel for c in self.conf_with[i])):
                    sel.add(i)
                    cost += self.cost[i]
                    power += self.power[i]
        y = np.zeros_like(x)
        y[list(sel)] = 1
        return y

    def __call__(self, X):
        return np.array([self.repair_one(x) for x in X], dtype=np.int8)


# ---------- штраф ----------
def penalized_fitness(inst, X, lam):
    """Минимизируемый фитнес: −V(x) + λ·(Δбюджет + Δмощность + #конфликтов + #зависимостей)."""
    v = violations(inst, X)
    return -(X @ inst.value) + lam * total_violation(v), total_violation(v) == 0

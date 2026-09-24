"""ЛР4: деревья выражений в префиксной записи, операторы ГП, компиляция в numpy-выражение.

Узел — кортеж: ("f", имя) — функция; ("x", i) — переменная; ("c", значение) — константа.
Дерево — список узлов в префиксном порядке; поддерево с корнем i занимает отрезок [i, subtree_end(i)).
Любой оператор заменяет целое поддерево целым поддеревом, поэтому потомки синтаксически корректны
по построению; ограничения глубины и размера проверяются, при нарушении возвращается копия родителя.
"""
import numpy as np

ARITY = {"add": 2, "sub": 2, "mul": 2, "div": 2, "sin": 1, "cos": 1, "exp": 1, "log": 1}
INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


# ---------- защищённые операции (определены на всей числовой оси) ----------
def _div(a, b):
    with np.errstate(all="ignore"):
        return np.where(np.abs(b) > 1e-9, a / np.where(b == 0, 1.0, b), 1.0)


def _log(a):
    with np.errstate(all="ignore"):
        return np.where(np.abs(a) > 1e-9, np.log(np.abs(np.where(a == 0, 1.0, a))), 0.0)


def _exp(a):
    return np.exp(np.clip(a, -50.0, 20.0))


NAMESPACE = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": _div,
             "sin": np.sin, "cos": np.cos, "exp": _exp, "log": _log}


def arity(node):
    return ARITY[node[1]] if node[0] == "f" else 0


def subtree_end(tree, i):
    need, j = 1, i
    while need:
        need += arity(tree[j]) - 1
        j += 1
    return j


def depth(tree):
    """Глубина (корень — глубина 0), один проход по префиксной записи."""
    best, stack = 0, []
    for node in tree:
        d = stack.pop() if stack else 0
        best = max(best, d)
        stack.extend([d + 1] * arity(node))
    return best


class Language:
    """Функциональное и терминальное множества; генерация случайных деревьев."""

    def __init__(self, functions, n_vars, p_const=0.3):
        self.functions = list(functions)
        self.n_vars = n_vars
        self.p_const = p_const

    def terminal(self, rng):
        if rng.random() < self.p_const:
            return ("c", round(float(rng.uniform(-1, 1)), 3))
        return ("x", int(rng.integers(self.n_vars)))

    def function(self, rng, arity_=None):
        pool = self.functions if arity_ is None else [f for f in self.functions if ARITY[f] == arity_]
        return ("f", pool[int(rng.integers(len(pool)))])

    def random_tree(self, rng, max_depth, method):
        """full — все листья на глубине max_depth; grow — листья на любой глубине."""
        out = []

        def build(d):
            n_term = self.n_vars + 1
            if d == max_depth or (method == "grow" and d > 0
                                  and rng.random() < n_term / (n_term + len(self.functions))):
                out.append(self.terminal(rng))
                return
            f = self.function(rng)
            out.append(f)
            for _ in range(ARITY[f[1]]):
                build(d + 1)

        build(0)
        return out

    def ramped_half_and_half(self, rng, n, min_depth, max_depth):
        depths = np.arange(min_depth, max_depth + 1)
        return [self.random_tree(rng, int(depths[k % len(depths)]), "full" if k % 2 else "grow")
                for k in range(n)]


def _pick(rng, tree, p_internal=0.9):
    """Точка вариации: с вероятностью 0.9 — внутренний узел (Koza), иначе лист."""
    internal = [i for i, nd in enumerate(tree) if nd[0] == "f"]
    leaves = [i for i, nd in enumerate(tree) if nd[0] != "f"]
    pool = internal if internal and rng.random() < p_internal else leaves
    return pool[int(rng.integers(len(pool)))]


def _ok(tree, max_depth, max_size):
    return len(tree) <= max_size and depth(tree) <= max_depth


def subtree_crossover(rng, p1, p2, max_depth, max_size):
    i, j = _pick(rng, p1), _pick(rng, p2)
    ie, je = subtree_end(p1, i), subtree_end(p2, j)
    c1 = p1[:i] + p2[j:je] + p1[ie:]
    c2 = p2[:j] + p1[i:ie] + p2[je:]
    return (c1 if _ok(c1, max_depth, max_size) else list(p1),
            c2 if _ok(c2, max_depth, max_size) else list(p2))


def subtree_mutation(rng, tree, lang, max_depth, max_size, new_depth=4):
    i = _pick(rng, tree, p_internal=0.5)
    child = tree[:i] + lang.random_tree(rng, int(rng.integers(1, new_depth + 1)), "grow") + \
        tree[subtree_end(tree, i):]
    return child if _ok(child, max_depth, max_size) else list(tree)


def point_mutation(rng, tree, lang, p_node=None):
    """Каждый узел с вероятностью 1/size заменяется узлом той же арности (константа — сдвигается)."""
    p_node = p_node or 1.0 / len(tree)
    out = list(tree)
    for k, nd in enumerate(out):
        if rng.random() < p_node:
            if nd[0] == "f":
                out[k] = lang.function(rng, ARITY[nd[1]])
            elif nd[0] == "c" and rng.random() < 0.5:
                out[k] = ("c", round(nd[1] + float(rng.normal(0, 0.1)), 3))
            else:
                out[k] = lang.terminal(rng)
    return out


# ---------- представление ----------
def to_code(tree):
    """Префиксное дерево -> строка вызовов numpy (ключ кэша и код для eval)."""
    def rec(i):
        nd = tree[i]
        if nd[0] == "x":
            return f"x{nd[1]}", i + 1
        if nd[0] == "c":
            return repr(nd[1]), i + 1
        args, j = [], i + 1
        for _ in range(ARITY[nd[1]]):
            a, j = rec(j)
            args.append(a)
        return f"{nd[1]}({', '.join(args)})", j
    return rec(0)[0]


def to_infix(tree):
    """Читаемая запись формулы."""
    def rec(i):
        nd = tree[i]
        if nd[0] == "x":
            return f"x{nd[1]}", i + 1
        if nd[0] == "c":
            return f"{nd[1]:g}", i + 1
        args, j = [], i + 1
        for _ in range(ARITY[nd[1]]):
            a, j = rec(j)
            args.append(a)
        if nd[1] in INFIX:
            return f"({args[0]} {INFIX[nd[1]]} {args[1]})", j
        return f"{nd[1]}({args[0]})", j
    s = rec(0)[0]
    return s[1:-1] if s.startswith("(") and s.endswith(")") else s


def predict(tree, X):
    ns = dict(NAMESPACE, **{f"x{i}": X[:, i] for i in range(X.shape[1])})
    with np.errstate(all="ignore"):
        out = eval(to_code(tree), {"__builtins__": {}}, ns)  # код построен только из собственных токенов
    return np.broadcast_to(np.asarray(out, float), (len(X),)).copy()

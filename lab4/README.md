# ЛР4: символьная регрессия генетическим программированием с контролем разрастания (вариант 5)

Заявка на тему: [`PROPOSAL.md`](PROPOSAL.md). Сравниваются 4 механизма контроля bloat при одинаковом языке, операторах
и бюджете 20 000 вычислений ФФ, по 30 запусков на 2 эталонных задачах (Nguyen-7, Pagie-1):

| вариант | конфиг | механизм |
|---|---|---|
| A | `gp_depth.toml` | только предел глубины 17 / размера 500 (эталон) |
| B | `gp_parsimony.toml` | фитнес NRMSE + c·size, c = 1e-3 |
| C | `gp_tarpeian.toml` | тарпейский метод, p = 0.6 |
| D | `gp_nsga2.toml` | NSGA-II по (NRMSE, size) из `evo_core/moo.py` |

## Запуск (из корня репозитория; установка — см. `lab1/README.md`)
```bash
# данные (уже лежат в data/, пересоздаются тем же seed)
.venv/bin/python -m lab4.data --seed 2026

# подбор c и p на seed 100..102, ~1.5 мин
.venv/bin/python -m lab4.tune

# полный набор: 4 варианта × 2 задачи × 30 запусков, ~10 мин на слабой машине
.venv/bin/python -m lab4.main --config lab4/configs/gp_depth.toml lab4/configs/gp_parsimony.toml lab4/configs/gp_tarpeian.toml lab4/configs/gp_nsga2.toml

# одна серия со своим seed
.venv/bin/python -m lab4.main --config lab4/configs/gp_nsga2.toml --seed 42 --runs 30

# статистика (Холм, Â12), формулы, графики; медианные запуски пересчитываются для проверки воспроизводимости
.venv/bin/python -m lab4.analyze --plots

.venv/bin/python -m pytest lab4
```

## Файлы
- `tree.py` — префиксные деревья, защищённые операции, ramped half-and-half, кроссовер и мутации поддеревьев, точечная мутация, компиляция в numpy-выражение.
- `gp.py` — цикл ГП, линейное масштабирование, кэш фитнеса, 4 режима контроля bloat, выбор модели по validation.
- `data.py` — задачи, шум 5 %, разбиение train/val/test 60/20/20 и точки экстраполяции.
- `results/`: `<задача>_<вариант>_summary.csv` (по запуску: NRMSE train/val/test/extrap, размер, глубина, вычисленные узлы, время, формула), `*_gens.csv` (динамика), `*_fronts.csv` (фронты D), `tuning.csv`, `summary.md`, `*.png`.

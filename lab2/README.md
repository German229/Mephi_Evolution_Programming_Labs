# ЛР2: ГА для выбора комплекта лабораторного оборудования (вариант 5)

Бинарная строка длины n = 40 (1 — прибор закупается). Максимизируется суммарная полезность при ограничениях:
бюджет Σc ≤ B, потребляемая мощность Σw ≤ P, несовместимые пары приборов, зависимости «A требует B».
Экземпляр: `data/instance.json` (40 приборов, 8 зависимостей, 12 несовместимостей; числа сгенерированы с seed 2026).

## Установка
```bash
python3 -m venv .venv                          # Python >= 3.11
.venv/bin/pip install -r requirements.txt      # numpy, matplotlib, pytest
```

## Запуск (из корня репозитория)
```bash
# экземпляр данных (уже лежит в data/, пересоздаётся тем же seed)
.venv/bin/python -m lab2.generate_data --seed 2026

# точный оптимум своим методом ветвей и границ, ~0.2 с
.venv/bin/python -m lab2.exact

# все серии (8 конфигураций × 20 запусков), ~40 с
.venv/bin/python -m lab2.main --config lab2/configs/*.toml

# одна серия со своим seed
.venv/bin/python -m lab2.main --config lab2/configs/ga_repair.toml --seed 42 --runs 20

# сводка, U-тесты, графики; примеры решений и лучший комплект
.venv/bin/python -m lab2.analyze --plots
.venv/bin/python -m lab2.examples

.venv/bin/python -m pytest lab2
```

## Конфигурации (`configs/`)
| файл | назначение |
|---|---|
| `ga_repair.toml` | база: ремонт + дозаполнение, равномерный кроссовер, bitflip |
| `ga_penalty.toml` | штраф λ = 300 вместо ремонта (сравнение способов учёта ограничений) |
| `ga_penalty_low.toml` | штраф λ = 50 (два коэффициента штрафа) |
| `ga_penalty_swap.toml` | штраф λ = 300 + мутация-обмен (сравнение операторов) |
| `ga_repair_swap.toml` | ремонт + мутация-обмен |
| `ga_repair_nofill.toml` | ремонт без жадного дозаполнения (абляция) |
| `random_feasible.toml` | случайный допустимый поиск, тот же бюджет 20 000 |
| `greedy.toml` | жадная конструктивная эвристика |

## Файлы
- `problem.py` — модель, нарушения ограничений, допустимость.
- `operators.py` — кроссоверы (равномерный, двухточечный), мутации (bitflip, swap), ремонт, штраф.
- `ga.py` — цикл ГА, случайный допустимый поиск, жадная эвристика.
- `exact.py` — ветви и границы (эталон качества).
- `main.py`, `analyze.py`, `examples.py`, `generate_data.py`.
- `results/`: `*_summary.csv` (по запуску: полезность, допустимость, стоимость, мощность, строка x), `*_gens.csv` (агрегаты по поколениям), `summary.md`, `examples.md`, `exact.json`, `*.png`.

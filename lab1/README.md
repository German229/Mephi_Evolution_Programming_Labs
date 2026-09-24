# ЛР1: вещественный ГА для функции Гриванка (вариант 5, d = 10)

$f(x) = 1 + \frac{1}{4000}\sum x_i^2 - \prod \cos(x_i/\sqrt{i})$, $x_i \in [-600, 600]$, минимум $f(0)=0$.

## Установка (из корня репозитория)
```bash
/opt/homebrew/bin/python3.12 -m venv .venv   # подойдёт любой Python ≥ 3.11 (нужен tomllib)
.venv/bin/pip install -r requirements.txt
```

## Запуск
```bash
# одна серия (20 запусков, seed 1000..1019), ~1 с
.venv/bin/python -m lab1.main --config lab1/configs/ga_base.toml

# все эксперименты ЛР1, ~5 с
.venv/bin/python -m lab1.main --config lab1/configs/*.toml

# свой seed / число запусков
.venv/bin/python -m lab1.main --config lab1/configs/ga_base.toml --seed 42 --runs 30

# сводная таблица, U-тесты и графики
.venv/bin/python -m lab1.analyze --plots

# предварительная развёртка σ на seed 0..9 (~2 с)
.venv/bin/python -m lab1.sweep

# тесты
.venv/bin/python -m pytest
```

## Конфигурации (`configs/`)
| файл | назначение | отличие от `ga_base` |
|---|---|---|
| `ga_base.toml` | базовый ГА | — |
| `ga_sigma_large.toml` | эксперимент 3 (один фактор) | `sigma_frac = 0.1` |
| `random_search.toml` | эксперимент 4 | `algorithm = "random"` |
| `ga_adaptive.toml` | доп. задание | `adaptive = true` |
| `ga_adaptive_pm03.toml` | доп. задание, настроенный | `adaptive = true`, `p_mutation = 0.3` |

## Результаты (`results/`)
- `<tag>_summary.csv` — по строке на запуск: seed, лучшее f, вызовы ФФ, время, допустимость, лучший x.
- `<tag>_gens.csv` — агрегаты по поколениям (каждое 5-е): best-so-far, лучшее/среднее/медиана популяции, медиана σ.
- `<tag>_config.json` — фактически использованные параметры.
- `summary.md` — таблица и U-тесты; `*.png` — графики.

## Файлы
- `operators.py` — функция Гриванка, BLX-α, равномерный кроссовер, гауссова мутация, самоадаптация σ.
- `ga.py` — цикл ГА и случайный поиск с единым учётом бюджета.
- `main.py` — серии запусков; `analyze.py` — статистика и графики.
- Общее ядро `../evo_core`: инициализация и учёт бюджета, селекция, элитизм, отражение от границ, статистика, графики, CLI.

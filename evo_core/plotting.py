"""Графики сходимости по серии запусков (matplotlib, без интерактива)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Категориальные цвета в фиксированном порядке (не циклически); текст — нейтральные чернила.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SURFACE = "#1f1f1e", "#5f5e58", "#e4e3dd", "#fcfcfb"
FLOOR = 1e-16  # нижняя граница для логарифмической шкалы


def _style(ax, xlabel, ylabel, log=True):
    ax.set_facecolor(SURFACE)
    if log:
        ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def _new(figsize=(8, 4.5), ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=figsize, facecolor=SURFACE)
    return fig, axes


def _save(fig, ax, title, path):
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def band_plot(ax, x, M, color, label, minmax=False, log=True, fmt="{:.1e}"):
    """Медиана по запускам + межквартильный диапазон (и, опционально, min–max). NaN — «ещё нет значения»."""
    if log:
        M = np.maximum(M, FLOOR)
    with np.errstate(all="ignore"):
        q25, med, q75 = np.nanpercentile(M, [25, 50, 75], axis=0)
    if minmax:
        ax.fill_between(x, np.nanmin(M, 0), np.nanmax(M, 0), color=color, alpha=0.10, linewidth=0,
                        label=f"{label}: min–max")
    ax.fill_between(x, q25, q75, color=color, alpha=0.25, linewidth=0, label=f"{label}: IQR")
    ax.plot(x, med, color=color, linewidth=2, label=f"{label}: медиана")
    ax.annotate(fmt.format(med[-1]), (x[-1], med[-1]), xytext=(4, 0), textcoords="offset points",
                color=INK_2, fontsize=8, va="center")


def convergence(x, M, title, path, xlabel="поколение", ylabel="лучшее f (best-so-far)", log=True,
                fmt="{:.1e}", ref=None):
    fig, ax = _new()
    band_plot(ax, x, M, SERIES[0], "серия", minmax=True, log=log, fmt=fmt)
    with np.errstate(all="ignore"):
        mean = np.nanmean(np.maximum(M, FLOOR) if log else M, axis=0)
    ax.plot(x, mean, color=INK_2, linewidth=1.5, linestyle="--", label="серия: среднее")
    _reference(ax, ref)
    _style(ax, xlabel, ylabel, log=log)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _save(fig, ax, title, path)


def _reference(ax, ref):
    """ref = (значение, подпись): горизонтальная опорная линия, например известный оптимум."""
    if ref is not None:
        ax.axhline(ref[0], color=INK_2, linewidth=1, linestyle="--")
        ax.annotate(ref[1], (0, ref[0]), xycoords=("axes fraction", "data"), xytext=(4, 4),
                    textcoords="offset points", color=INK_2, fontsize=8)


def compare(curves, title, path, xlabel="вычисления ФФ", ylabel="лучшее f (best-so-far)", log=True,
            fmt="{:.1e}", ref=None, ylim=None):
    """curves: список (label, x, M) — каждая серия своим цветом из SERIES по порядку."""
    fig, ax = _new()
    for (label, x, M), c in zip(curves, SERIES):
        band_plot(ax, x, M, c, label, log=log, fmt=fmt)
    _reference(ax, ref)
    _style(ax, xlabel, ylabel, log=log)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _save(fig, ax, title, path)


def compare_with_panel(curves, panel, title, panel_ylabel, path, xlabel="вычисления ФФ"):
    """Две малые мультипликации с общей осью x: сходимость и вспомогательная величина."""
    fig, (a1, a2) = _new(figsize=(11, 4.5), ncols=2)
    for (label, x, M), c in zip(curves, SERIES):
        band_plot(a1, x, M, c, label)
    _style(a1, xlabel, "лучшее f (best-so-far)")
    a1.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    for (label, x, M), c in zip(panel, SERIES):
        a2.plot(x, np.median(np.maximum(M, FLOOR), axis=0), color=c, linewidth=2, label=label)
    _style(a2, xlabel, panel_ylabel)
    a2.legend(frameon=False, fontsize=8, labelcolor=INK)
    a2.set_title("медиана по серии", loc="left", color=INK_2, fontsize=10)
    _save(fig, a1, title, path)


def final_boxes(groups, title, path, ylabel="итоговое лучшее f", log=True, ref=None):
    """groups: список (label, values). Ящики + точки отдельных запусков."""
    fig, ax = _new(figsize=(8, 4.5))
    data = [np.asarray(v, float) for _, v in groups]
    data = [np.maximum(v, FLOOR) if log else v[~np.isnan(v)] for v in data]
    ax.boxplot(data, widths=0.5, showfliers=False,
               medianprops={"color": INK, "linewidth": 2},
               boxprops={"color": INK_2}, whiskerprops={"color": INK_2}, capprops={"color": INK_2})
    rng = np.random.default_rng(0)
    for i, v in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, v.size), v, s=18, color=SERIES[0], alpha=0.7,
                   edgecolors=SURFACE, linewidths=0.8, zorder=3)
    ax.set_xticks(range(1, len(groups) + 1), [g for g, _ in groups], fontsize=9)
    _reference(ax, ref)
    _style(ax, "", ylabel, log=log)
    _save(fig, ax, title, path)

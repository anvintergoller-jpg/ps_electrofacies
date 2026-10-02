# -*- coding: utf-8 -*-
"""
Справочный рисунок: типы формы аномалии ПС (v3.0).
«Название — схематичный профиль — структура».

Сохраняется как PNG. Используется геологом как «шпаргалка» рядом
с планшетом: видно, что означают типы bell/funnel/cylinder/...,
и какая у них структура элементов.

Палитра FORM_COLORS — общая с well_log.py. Если поменять здесь,
поменяется и цвет эталонных кривых на планшете.

v3.0. Состав форм (см. docs/CLASSIFICATION_RULES_v3.md, §7.6):
    bell, funnel, v-shape, trapezoid-top, trapezoid-bottom,
    trapezoid-middle, cylinder, unknown-shape.
Убраны legacy v2.x: symmetric, uncertain.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from src.domain.reference_curve import parabola_through_points


# Цвета по типу формы. Используются и в легенде, и при наложении
# эталонных кривых на планшете.
#
# Нумерация цветов следует порядку форм в FORM_CODES (petrel_export.py):
#   1 bell, 2 funnel, 3 v-shape,
#   4 trapezoid-top, 5 trapezoid-bottom, 6 trapezoid-middle,
#   7 cylinder, 8 unknown-shape.
FORM_COLORS = {
    "bell":             "#1f77b4",   # синий
    "funnel":           "#ff7f0e",   # оранжевый
    "v-shape":          "#8c564b",   # коричневый
    "trapezoid-top":    "#d62728",   # красный
    "trapezoid-bottom": "#bcbd22",   # жёлто-зелёный
    "trapezoid-middle": "#e377c2",   # розовый
    "cylinder":         "#2ca02c",   # зелёный
    "unknown-shape":    "#7f7f7f",   # серый
    "non_reservoir":    "#cccccc",   # светло-серый (для колонки интервалов)
}


# Схематичные профили aSP (относительная глубина → aSP).
# t = 0 — кровля, t = 1 — подошва.
# aSP: 0 — глина, 1 — песок. Ось инвертирована при отрисовке
# (1 слева, 0 справа) — согласовано с сырой SP на планшете.
_SCHEMAS = {
    # --- Треугольники (плато нет) ---
    "bell": {
        # Плавный вход сверху, резкий выход снизу.
        # Лучший коллектор — внизу.
        "t": [0.0, 0.5, 0.5, 1.0],
        "sp": [0.15, 0.85, 0.85, 0.85],
        "title": "bell — колокол",
        "cond": "плавный вход сверху, резкий выход снизу",
    },
    "funnel": {
        # Резкий вход сверху, плавный выход снизу.
        # Лучший коллектор — вверху.
        "t": [0.0, 0.5, 0.5, 1.0],
        "sp": [0.85, 0.85, 0.85, 0.15],
        "title": "funnel — воронка",
        "cond": "резкий вход сверху, плавный выход снизу",
    },
    "v-shape": {
        # Обе линии плавные, максимум в середине.
        "t": [0.0, 0.5, 1.0],
        "sp": [0.15, 0.85, 0.15],
        "title": "v-shape — V-форма",
        "cond": "обе границы плавные, максимум в середине",
    },

    # --- Трапеции (плато есть) ---
    "trapezoid-top": {
        # Плавная кровля + плато + резкая подошва.
        "t": [0.0, 0.5, 0.9, 1.0],
        "sp": [0.15, 0.85, 0.85, 0.15],
        "title": "trapezoid-top",
        "cond": "плавный вход, плато, резкий выход",
    },
    "trapezoid-bottom": {
        # Резкая кровля + плато + плавная подошва.
        "t": [0.0, 0.1, 0.5, 1.0],
        "sp": [0.15, 0.85, 0.85, 0.15],
        "title": "trapezoid-bottom",
        "cond": "резкий вход, плато, плавный выход",
    },
    "trapezoid-middle": {
        # Обе границы плавные + плато в середине.
        "t": [0.0, 0.25, 0.75, 1.0],
        "sp": [0.15, 0.85, 0.85, 0.15],
        "title": "trapezoid-middle",
        "cond": "обе границы плавные, плато в середине",
    },

    # --- Прямоугольник ---
    "cylinder": {
        # Обе границы мгновенные, длинное плато.
        "t": [0.0, 0.05, 0.95, 1.0],
        "sp": [0.15, 0.85, 0.85, 0.15],
        "title": "cylinder — цилиндр",
        "cond": "обе границы мгновенные, длинное плато",
    },

    # --- Осмысленный сигнал ---
    "unknown-shape": {
        # Зигзаг — форма не подошла ни под один шаблон.
        "t": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "sp": [0.20, 0.55, 0.30, 0.65, 0.45, 0.35],
        "title": "unknown-shape",
        "cond": "структура не подошла ни под один шаблон",
    },
}


def plot_form_legend(output_path, cutoff):
    """
    Нарисовать справочный PNG с типами формы v3.0.

    Ось X — aSP (1 = песок слева, 0 = глина справа).
    Та же полярность, что на правой оси трека SP на планшете,
    поэтому геолог может сверять планшет и легенду напрямую,
    без мысленного зеркалирования.

    Параметры
    ---------
    output_path : str | Path
        Куда сохранить PNG.
    cutoff : float
        Порог коллектора по aSP (обычно 0.4). Сейчас не используется
        в отрисовке (в легенде порог не несёт смысла), но сохранён
        в сигнатуре для совместимости с run_pipeline.
    """
    # Сетка 4 × 2 = 8 панелей: ровно под 8 форм v3.0.
    fig, axes = plt.subplots(
        nrows=2, ncols=4,
        figsize=(14, 8),
        gridspec_kw={"hspace": 0.55, "wspace": 0.35},
    )
    axes_flat = axes.ravel()

    # Порядок вывода: сверху — треугольники и одна трапеция,
    # снизу — остальные трапеции и cylinder + unknown.
    order = [
        "bell", "funnel", "v-shape", "trapezoid-top",
        "trapezoid-bottom", "trapezoid-middle", "cylinder",
        "unknown-shape",
    ]

    for ax, key in zip(axes_flat, order):
        schema = _SCHEMAS[key]
        color = FORM_COLORS[key]

        # Интерполируем схему. Для сглаженных схем (parabola) —
        # гладкая кривая, для остальных — кусочно-линейная.
        t_dense = np.linspace(0.0, 1.0, 200)
        t_ops = schema["t"]
        sp_ops = schema["sp"]

        if schema.get("curve") == "parabola":
            sp_dense = parabola_through_points(
                t_dense, t_ops=t_ops, y_ops=sp_ops,
            )
            sp_dense = np.clip(sp_dense, 0.0, 1.0)
        else:
            sp_dense = np.interp(t_dense, t_ops, sp_ops)

        # Заливка между кривой и глинистой базой (x = 0).
        ax.fill_betweenx(
            t_dense,
            sp_dense,
            0.0,
            color=color,
            alpha=0.20,
        )

        # Сама кривая — поверх заливки.
        ax.plot(sp_dense, t_dense, color=color, lw=2.4)

        # Ось инвертирована: 1.0 (песок) слева, 0.0 (глина) справа.
        ax.set_xlim(1.0, 0.0)
        ax.set_ylim(1.0, 0.0)
        ax.set_xticks([1.0, 0.0])
        ax.set_xticklabels(
            ["1.0\n(песок)", "0\n(глина)"],
            fontsize=8,
        )
        ax.set_yticks([])
        ax.set_title(schema["title"], fontsize=11, color=color)
        ax.grid(alpha=0.25)

        # Условие — под графиком.
        ax.text(
            0.5, -0.18, schema["cond"],
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=8, family="monospace",
        )

    fig.suptitle(
        "Типы формы аномалий ПС (v3.0): название — профиль aSP — структура\n"
        "(1 = песок, 0 = глина — как на треке SP в Petrel)",
        fontsize=13,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Легенда форм сохранена: {output_path.resolve()}")
    except OSError as e:
        alt = output_path.with_name(
            output_path.stem + "_new" + output_path.suffix
        )
        fig.savefig(alt, dpi=150, bbox_inches="tight")
        print(f"  [!] Не удалось перезаписать {output_path.name}: {e}")
        print(f"  [!] Сохранено как {alt.name} — "
              f"закройте старый файл и перезапустите пайплайн.")

    plt.close(fig)


__all__ = ["plot_form_legend", "FORM_COLORS"]
# -*- coding: utf-8 -*-
"""
Справочный рисунок: 8 типов формы аномалии ПС.
«Название — схематичный профиль — условие».

Сохраняется как PNG. Используется геологом как «шпаргалка» рядом
с планшетом: видно, что означают типы bell/funnel/cylinder/...,
и по какому условию они выделены.

Палитра FORM_COLORS — общая с well_log.py. Если поменять здесь,
поменяется и цвет эталонных кривых на планшете.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from src.domain.reference_curve import parabola_through_points


# Цвета по типу формы. Используются и в легенде, и при наложении
# эталонных кривых на планшете.
FORM_COLORS = {
    "bell":             "#1f77b4",   # синий
    "funnel":           "#ff7f0e",   # оранжевый
    "cylinder":         "#2ca02c",   # зелёный
    "trapezoid-middle": "#e377c2",   # розовый
    "symmetric":        "#9467bd",   # фиолетовый
    "v-shape":          "#8c564b",   # коричневый
    "uncertain":        "#7f7f7f",   # серый
    "non_reservoir":    "#cccccc",   # светло-серый (для колонки интервалов)
}


# Схематичные профили SP_norm (относительная глубина → SP_norm).
# t = 0 — кровля, t = 1 — подошва.
# SP_norm: 0 — чистый песчаник, 1 — чистая глина.
# Та же полярность, что на планшете (трек SP, правая ось).
# Схематичные профили SP_norm (относительная глубина → SP_norm).
# t = 0 — кровля, t = 1 — подошва.
# SP_norm: 0 — песчаник, 1 — глина.
_SCHEMAS = {
    "bell": {
        # Глина сверху → линейный спад → песчаное плато снизу.
        "t": [0.0, 0.25, 0.75, 1.0],
        "sp": [0.85, 0.85, 0.15, 0.15],
        "title": "bell — колокол",
        "cond": "глина сверху → плато песка снизу",
    },
    "funnel": {
        # Зеркало bell: плато песка сверху → линейный подъём → глина снизу.
        "t": [0.0, 0.25, 0.75, 1.0],
        "sp": [0.15, 0.15, 0.85, 0.85],
        "title": "funnel — воронка",
        "cond": "плато песка сверху → глина снизу",
    },
    "cylinder": {
        # Ровное плато на песчаном уровне.
        "t": [0.0, 0.5, 1.0],
        "sp": [0.15, 0.15, 0.15],
        "title": "cylinder — цилиндр",
        "cond": "ровное плато на песчаном уровне",
    },
    "trapezoid-middle": {
        # Симметричная трапеция: спад 25% → плато 50% → подъём 25%.
        "t": [0.0, 0.25, 0.75, 1.0],
        "sp": [0.85, 0.15, 0.15, 0.85],
        "title": "trapezoid-middle",
        "cond": "песчаное плато в середине, глина по краям",
    },
    "symmetric": {
        # Парабола через три точки — гладкая чаша.
        "t": [0.0, 0.5, 1.0],
        "sp": [0.85, 0.15, 0.85],
        "curve": "parabola",
        "title": "symmetric — симметричная",
        "cond": "плавная чаша, экстремум в центре (парабола)",
    },
    "v-shape": {
        # Две прямые через три точки — острый угол в центре.
        "t": [0.0, 0.5, 1.0],
        "sp": [0.85, 0.15, 0.85],
        "title": "v-shape — V-форма",
        "cond": "острый угол, экстремум в центре (две прямые)",
    },
    "uncertain": {
        # Рваная кривая — ничего конкретного.
        "t": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "sp": [0.80, 0.45, 0.70, 0.35, 0.55, 0.65],
        "title": "uncertain",
        "cond": "ни одно правило не сработало",
    },
}


def plot_form_legend(output_path):
    """
    Нарисовать справочный PNG с 8 типами формы.

    Ось X — SP_norm (0 = песок слева, 1 = глина справа).
    Та же полярность, что на правой оси трека SP на планшете,
    поэтому геолог может сверять планшет и легенду напрямую,
    без мысленного зеркалирования.

    Параметры
    ---------
    output_path : str | Path
        Куда сохранить PNG (обычно results/plots/form_legend.png).
    """
    # Сетка 4 в ширину × 2 в высоту = 8 панелей.
    fig, axes = plt.subplots(
        nrows=2, ncols=4,
        figsize=(14, 8),
        gridspec_kw={"hspace": 0.55, "wspace": 0.35},
    )
    axes_flat = axes.ravel()

    # Порядок вывода. Сетка 2×4, заполняется слева-направо, сверху-вниз.
    # Верхний ряд — формы с преобладающим наклоном.
    # Нижний ряд — формы-плато и особые случаи.
        # 7 форм + один резервный слот в сетке 2×4.
    # Верхний ряд — формы с наклоном и с экстремумом в центре.
    # Нижний ряд — плато и особые.
    order = [
        "bell", "funnel", "v-shape",
        "symmetric", "cylinder", "trapezoid-middle",
        "uncertain", None,   # None → пустая панель
    ]

    # Порог коллектора по SP_norm — вертикальная линия.
    # Совпадает с segmentation.cutoff в config.yaml.
    SP_CUTOFF = 0.6

    for ax, key in zip(axes_flat, order):
        # Пустая панель (резервный слот).
        if key is None:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("(резерв)", fontsize=10, color="grey")
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        schema = _SCHEMAS[key]
        color = FORM_COLORS[key]

                # Интерполируем сглаженную схему.
        # Для symmetric — парабола через три точки (гладкая чаша),
        # для остальных — кусочно-линейная (угловатая схема).
        t_dense = np.linspace(0.0, 1.0, 200)
        t_ops = schema["t"]
        sp_ops = schema["sp"]

        if schema.get("curve") == "parabola":
            # Парабола через три точки. Никаких плато на краях —
            # кривая идёт от t=0 до t=1 целиком.
            sp_dense = parabola_through_points(
                t_dense, t_ops=t_ops, y_ops=sp_ops,
            )
            # На всякий случай обрезаем к [0, 1].
            sp_dense = np.clip(sp_dense, 0.0, 1.0)
        else:
            sp_dense = np.interp(t_dense, t_ops, sp_ops)

                # Ось X — SP_norm (0..1, слева песок, справа глина).
        # Ось Y — относительная глубина (0..1, сверху кровля).

        # Заливка между кривой и глинистой базой (x = 1).
        # Это классический вид трека SP: цветом подсвечена
        # «вылазка» кривой в песчаную сторону. Чем песчанее участок,
        # тем шире заливка. На глине кривая уходит к x = 1,
        # и заливка сужается почти до нуля.
        ax.fill_betweenx(
            t_dense,
            sp_dense,     # левая граница заливки — сама кривая
            1.0,          # правая граница — глинистая база
            color=color,
            alpha=0.20,
        )

        # Сама кривая — рисуем ПОСЛЕ заливки, чтобы она была сверху.
        ax.plot(sp_dense, t_dense, color=color, lw=2.4)

        # Вертикальная линия-порог коллектора SP_norm = 0.6.
        ax.axvline(x=SP_CUTOFF, color="grey", lw=0.6, ls=":", alpha=0.7)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0, 0.0)   # 0 вверху, 1 внизу — как на планшете
        ax.set_xticks([0.0, SP_CUTOFF, 1.0])
        ax.set_xticklabels(
            ["0\n(песок)", f"{SP_CUTOFF}", "1.0\n(глина)"],
            fontsize=8,
        )
        ax.set_yticks([])
        ax.set_title(schema["title"], fontsize=11, color=color)
        ax.grid(alpha=0.25)

        # Условие — под графиком
        ax.text(
            0.5, -0.18, schema["cond"],
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=8, family="monospace",
        )

    fig.suptitle(
        "Типы формы аномалий ПС: название — профиль SP_norm — условие\n"
        "(0 = песок, 1 = глина;  αПС = 1 − SP_norm;  "
        "сглаживание SP_norm = 5 точек)",
        fontsize=13,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # На Windows перезапись занятого файла падает с OSError 22.
    # Ловим и сохраняем рядом с суффиксом _new, чтобы не терять результат.
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
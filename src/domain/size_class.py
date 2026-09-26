# -*- coding: utf-8 -*-
"""
Категории размера интервала и положение в контейнере (шаг E).

См. docs/FEATURES.md, §5.4 (положение) и §7 (размер).

Две независимые характеристики каждого коллекторного интервала:

1. size_class — small / medium / large по ДОЛЕ интервала
   в мощности родительского пласта-контейнера:

       relative = thickness_interval_tvdss / thickness_container_tvdss
       small:  relative < 0.33
       medium: 0.33 .. 0.67
       large:  relative > 0.67

   Пороги ФИКСИРОВАННЫЕ (не перцентили выборки) — чтобы результаты
   были сопоставимы между скважинами: интервал, занимающий 40%
   мощности БС9 в скважине А, будет medium и в скважине Б,
   независимо от того, какие ещё интервалы там есть.

   Относительная доля естественно даёт перекос в small —
   коллектор обычно занимает меньшую часть пласта (остальное глина).
   Это ожидаемо и корректно.

2. position_in_container — top / middle / bottom по положению
   центроида интервала в родительском пласте-контейнере:

       centroid_relative = (top_container_tvdss - centroid_tvdss)
                           / (top_container_tvdss - bottom_container_tvdss)
       top:    < 0.33
       middle: 0.33 .. 0.67
       bottom: > 0.67

Обе — колонки в intervals.csv. Никаких отдельных таблиц:
все критерии относятся к аномалии внутри пласта (см. FEATURES.md, §5).

Особенности:
    • centroid_tvdss — среднее по ТОЧКАМ интервала, не по границам.
      Устойчивее для интервалов с неравномерным распределением точек.
    • size_class присваивается только reservoir-интервалам.
      У non_reservoir — "n/a".
    • Если контейнер не найден в layers (нет top/bottom_tvdss) —
      size_class = unknown, position_in_container = None.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def _centroid_tvdss(
    df: pd.DataFrame, top_md: float, bottom_md: float
) -> Optional[float]:
    """
    Средний TVDSS точек интервала.

    Параметры
    ---------
    df : pd.DataFrame — общий датасет (нужны depth_md, depth_tvdss)
    top_md, bottom_md : float — границы интервала по MD

    Возвращает
    ----------
    float или None — среднее depth_tvdss точек, попавших в интервал.
    """
    mask = (df["depth_md"] >= top_md) & (df["depth_md"] <= bottom_md)
    sub = df.loc[mask, "depth_tvdss"]
    if sub.empty:
        return None
    return float(sub.mean())


def _centroid_relative(
    centroid_tvdss: Optional[float],
    top_container_tvdss: float,
    bottom_container_tvdss: float,
) -> Optional[float]:
    """
    Относительное положение центроида в контейнере.

    TVDSS в Petrel-стиле: оба значения отрицательные, top > bottom.
    Тогда:
        0 = кровля контейнера,
        1 = подошва.
    """
    if centroid_tvdss is None:
        return None
    span = top_container_tvdss - bottom_container_tvdss
    if span <= 0:
        return None
    return (top_container_tvdss - centroid_tvdss) / span


def _position_category(
    centroid_relative: Optional[float],
    pos_top_max: float,
    pos_bottom_min: float,
) -> Optional[str]:
    """
    Категория положения: top / middle / bottom.

    top:    centroid_relative < pos_top_max
    middle: pos_top_max ≤ centroid_relative ≤ pos_bottom_min
    bottom: centroid_relative > pos_bottom_min
    """
    if centroid_relative is None:
        return None
    if centroid_relative < pos_top_max:
        return "top"
    if centroid_relative > pos_bottom_min:
        return "bottom"
    return "middle"


def _size_category(
    thickness_interval: float,
    thickness_container: float,
    rel_small_max: float,
    rel_large_min: float,
) -> str:
    """
    Категория размера интервала по его ДОЛЕ в мощности контейнера.

    relative = thickness_interval / thickness_container

    Пороги фиксированные (0.33 / 0.67 по умолчанию) — не перцентили
    выборки. Это обеспечивает сопоставимость между скважинами.
    """
    if thickness_container is None or thickness_container <= 0:
        return "unknown"

    relative = thickness_interval / thickness_container

    if relative < rel_small_max:
        return "small"
    if relative > rel_large_min:
        return "large"
    return "medium"


def classify_size_and_position(
    df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    layers: list[dict],
    cfg: dict,
) -> pd.DataFrame:
    """
    Обогатить intervals_df колонками:

        centroid_tvdss        — центр масс интервала в АО
        centroid_relative     — 0..1, положение в контейнере
        position_in_container — top / middle / bottom
        size_relative         — доля интервала в мощности контейнера
        size_class            — small / medium / large / unknown / n/a

    Параметры
    ---------
    df : pd.DataFrame
        Общий датасет (нужны depth_md, depth_tvdss).
    intervals_df : pd.DataFrame
        Результат сегментации + классификации формы.
        Нужны колонки: layer_name, kind, top_md, bottom_md, thickness_tvdss.
    layers : list[dict]
        Пласты-контейнеры из df.attrs["layers"].
        Нужны поля name, top_tvdss, bottom_tvdss, thickness_tvdss.
    cfg : dict
        Секция size_classification из config.yaml.

    Возвращает
    ----------
    pd.DataFrame — копия intervals_df с новыми колонками.
    """
    new_cols = [
        "centroid_tvdss", "centroid_relative",
        "position_in_container", "size_relative", "size_class",
    ]

    # Пустой вход — возвращаем пустой DataFrame с нужными колонками,
    # чтобы дальше по пайплайну не спотыкаться об отсутствие столбцов.
    if intervals_df.empty:
        out = intervals_df.copy()
        for col in new_cols:
            out[col] = pd.Series(dtype=object)
        return out

    # Параметры из конфига.
    rel_small_max = cfg.get("relative_small_max", 0.33)
    rel_large_min = cfg.get("relative_large_min", 0.67)
    pos_top_max = cfg.get("position_top_max", 0.33)
    pos_bottom_min = cfg.get("position_bottom_min", 0.67)

    # Словарь контейнеров: имя → (top_tvdss, bottom_tvdss, thickness_tvdss).
    # Пропускаем контейнеры без границ — тогда position_in_container
    # и size_class для их интервалов будут None/unknown.
    containers: dict[str, tuple] = {}
    for lay in layers:
        top = lay.get("top_tvdss")
        bottom = lay.get("bottom_tvdss")
        thickness = lay.get("thickness_tvdss")
        if top is None or bottom is None:
            continue
        if thickness is None:
            # Вычислим сами, если не пришло: top − bottom
            thickness = top - bottom
        containers[lay["name"]] = (top, bottom, thickness)

    rows: list[dict] = []

    for _, iv in intervals_df.iterrows():
        rec = iv.to_dict()

        # --- centroid_tvdss — среднее по точкам интервала -----------
        centroid = _centroid_tvdss(df, rec["top_md"], rec["bottom_md"])
        rec["centroid_tvdss"] = centroid

        # --- Данные родительского контейнера ------------------------
        top_c, bot_c, thick_c = containers.get(
            rec["layer_name"], (None, None, None)
        )

        # --- centroid_relative — положение в контейнере -------------
        if top_c is not None and bot_c is not None:
            cr = _centroid_relative(centroid, top_c, bot_c)
        else:
            cr = None
        rec["centroid_relative"] = cr

        # --- position_in_container ----------------------------------
        rec["position_in_container"] = _position_category(
            cr, pos_top_max, pos_bottom_min
        )

        # --- size_relative и size_class -----------------------------
        # Относительный размер интервала — доля в контейнере.
        # Считаем только для reservoir. Неколлекторам — n/a.
        if rec["kind"] != "reservoir":
            rec["size_relative"] = None
            rec["size_class"] = "n/a"
        else:
            if thick_c is not None and thick_c > 0:
                rel = rec["thickness_tvdss"] / thick_c
                rec["size_relative"] = rel
                rec["size_class"] = _size_category(
                    rec["thickness_tvdss"], thick_c,
                    rel_small_max, rel_large_min,
                )
            else:
                # Контейнер без мощности — посчитать нельзя.
                rec["size_relative"] = None
                rec["size_class"] = "unknown"

        rows.append(rec)

    # --- Собираем итоговый DataFrame, сохраняя порядок колонок ---
    base_cols = list(intervals_df.columns)
    out = pd.DataFrame(rows)[base_cols + new_cols]
    return out


__all__ = ["classify_size_and_position"]
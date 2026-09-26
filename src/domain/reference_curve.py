# -*- coding: utf-8 -*-
"""
Построение эталонной кривой SP_norm для коллекторного интервала.

Для каждого интервала есть три опорных значения SP_norm
(sp_top, sp_mid, sp_bot) в трёх относительных точках интервала:
    t_top = edge_frac / 2      (середина верхней зоны)
    t_mid = 0.5                (середина)
    t_bot = 1 − edge_frac / 2  (середина нижней зоны)

Геометрия эталона зависит от типа формы (см. build_reference_curve).
На планшете эталон рисуется штриховой линией поверх реальной SP_norm:
совпадение — классификация верна, расхождение — видно, что́ имел
в виду алгоритм и где он промахнулся.
"""

from __future__ import annotations

import numpy as np


def parabola_through_points(
    t: np.ndarray,
    t_ops: list[float],
    y_ops: list[float],
) -> np.ndarray:
    """
    Парабола через три точки (интерполяционный полином Лагранжа).

    y(t) = y1·L1(t) + y2·L2(t) + y3·L3(t),
    где Li — базисные полиномы Лагранжа степени 2.

    Параметры
    ---------
    t : np.ndarray
        Точки, в которых считаем y.
    t_ops : list[float]
        Три абсциссы опорных точек (должны быть различны).
    y_ops : list[float]
        Три ординаты опорных точек.

    Возвращает
    ----------
    np.ndarray той же длины, что t.
    """
    t1, t2, t3 = t_ops
    y1, y2, y3 = y_ops

    d12 = t1 - t2
    d13 = t1 - t3
    d23 = t2 - t3

    L1 = (t - t2) * (t - t3) / (d12 * d13)
    L2 = (t - t1) * (t - t3) / (-d12 * d23)
    L3 = (t - t1) * (t - t2) / (d13 * d23)

    return y1 * L1 + y2 * L2 + y3 * L3


def build_reference_curve(
    depth_tvdss: np.ndarray,
    top_tvdss: float,
    bottom_tvdss: float,
    sp_top: float,
    sp_mid: float,
    sp_bot: float,
    edge_frac: float,
    form_type: str = "bell",
) -> np.ndarray:
    """
    Построить эталонную кривую SP_norm для одного интервала.

    Работаем в SP_norm (0 = песок, 1 = глина) — та же полярность,
    что у сырой кривой SP на планшете.

    Шаблоны (t — относительная глубина, 0 кровля, 1 подошва):

        cylinder          — ровное плато на sp_mid
        bell              — плато sp_top (0..0.25), линейный переход,
                            плато sp_bot (0.75..1)
        funnel            — зеркало bell
        trapezoid-middle  — спад 0..0.25, плато sp_mid (0.25..0.75),
                            подъём 0.75..1
        symmetric         — парабола через три опоры, без плато
        v-shape           — две прямые через три опоры, острый угол
        остальные         — линейная интерполяция через 5 точек
                            (совместимость, устаревшие формы)

    Параметры
    ---------
    depth_tvdss : np.ndarray
        Глубины точек интервала (в АО).
    top_tvdss, bottom_tvdss : float
        Кровля / подошва интервала в АО.
    sp_top, sp_mid, sp_bot : float
        Опорные SP_norm из classify_form.
    edge_frac : float
        Доля мощности опорных зон (для «устаревшего» шаблона).
    form_type : str
        Тип формы (см. список шаблонов выше).

    Возвращает
    ----------
    np.ndarray SP_norm_ref той же длины, что depth_tvdss.
    Значения обрезаны до [0, 1].
    """
    # t = 0 в кровле, t = 1 в подошве.
    span = top_tvdss - bottom_tvdss
    if span <= 0:
        return np.full_like(depth_tvdss, np.nan, dtype=float)

    t = (top_tvdss - depth_tvdss) / span

    # --- cylinder: ровное плато на sp_mid -----------------------------
    if form_type == "cylinder":
        y = np.full_like(t, sp_mid, dtype=float)
        return np.clip(y, 0.0, 1.0)

    # --- bell: глина сверху, песок снизу ------------------------------
    if form_type == "bell":
        # Плато sp_top 0..0.25, линейный переход, плато sp_bot 0.75..1.
        xp = [0.0, 0.25, 0.75, 1.0]
        fp = [sp_top, sp_top, sp_bot, sp_bot]
        y = np.interp(t, xp, fp)
        return np.clip(y, 0.0, 1.0)

    # --- funnel: зеркало bell -----------------------------------------
    if form_type == "funnel":
        xp = [0.0, 0.25, 0.75, 1.0]
        fp = [sp_top, sp_top, sp_bot, sp_bot]
        y = np.interp(t, xp, fp)
        return np.clip(y, 0.0, 1.0)

    # --- trapezoid-middle: симметричная трапеция ----------------------
    if form_type == "trapezoid-middle":
        xp = [0.0, 0.25, 0.75, 1.0]
        fp = [sp_top, sp_mid, sp_mid, sp_bot]
        y = np.interp(t, xp, fp)
        return np.clip(y, 0.0, 1.0)

    # --- symmetric: парабола через три опоры --------------------------
    if form_type == "symmetric":
        y = parabola_through_points(
            t,
            t_ops=[0.0, 0.5, 1.0],
            y_ops=[sp_top, sp_mid, sp_bot],
        )
        return np.clip(np.asarray(y, dtype=float), 0.0, 1.0)

    # --- v-shape: две прямые через три опоры, острый угол -------------
    if form_type == "v-shape":
        xp = [0.0, 0.5, 1.0]
        fp = [sp_top, sp_mid, sp_bot]
        y = np.interp(t, xp, fp)
        return np.clip(y, 0.0, 1.0)

    # --- fallback: линейная интерполяция с плато на краях -------------
    # Для неизвестных форм (в т.ч. устаревших trapezoid-top/bottom) —
    # чтобы код не падал, если они случайно придут.
    t_top = edge_frac / 2.0
    t_bot = 1.0 - edge_frac / 2.0
    xp = [0.0, t_top, 0.5, t_bot, 1.0]
    fp = [sp_top, sp_top, sp_mid, sp_bot, sp_bot]
    y = np.interp(t, xp, fp)
    y = np.asarray(y, dtype=float)
    y = np.where(t <= t_top, sp_top, y)
    y = np.where(t >= t_bot, sp_bot, y)
    return np.clip(y, 0.0, 1.0)


__all__ = ["build_reference_curve", "parabola_through_points"]
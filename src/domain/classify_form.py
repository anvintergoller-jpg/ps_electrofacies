# -*- coding: utf-8 -*-
"""
Классификация формы аномалии ПС для коллекторных интервалов (шаг D.3).

Источник правил: docs/CLASSIFICATION_RULES.md (версия 2.0).

Идея
----
На вход — массив SP_norm ТОЛЬКО для точек одного коллекторного
интервала (сверху вниз). На выходе — тип формы, уверенность
и короткое объяснение для геолога.

Шкала SP_norm: ~1.0 = глина, ~0.0 = песчаник.

Типы форм (7):
    cylinder          — ровное плато на песчаном уровне
    bell              — глина сверху, песок снизу
    funnel            — песок сверху, глина снизу
    trapezoid-middle  — глина по краям, широкое плато песка в центре
    symmetric         — глина по краям, плавная чаша песка (парабола)
    v-shape           — глина по краям, острый угол песка в центре
    uncertain         — не определено (в т.ч. возможное переслаивание)
    non_reservoir     — неколлектор

Ключевой признак frac_core — доля точек, где |SP_norm − sp_mid| < core_threshold.
Разделяет trapezoid-middle / symmetric / v-shape (у них одинаковые опоры,
но разная форма перехода).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Переиспользуем сглаживание из сегментации — не дублируем логику.
from src.domain.segmentation import smooth


# Версия правил. Поднимается вручную при изменении методики.
RULES_VERSION = "2.0"


# ===========================================================================
#  Параметры классификации
# ===========================================================================

@dataclass
class ClassificationParams:
    """
    Набор параметров классификации. Значения по умолчанию совпадают
    с секцией `classification` в config/config.yaml.
    """
    edge_frac: float = 0.15
    threshold_diff: float = 0.20
    plateau_threshold: float = 0.10
    slope_threshold: float = 0.20
    core_threshold: float = 0.05
    frac_core_trapezoid: float = 0.40
    frac_core_v_shape: float = 0.15
    min_points: int = 5
    reservoir_cutoff: float = 0.60
    smooth_window: int = 5
    rules_version: str = RULES_VERSION

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ClassificationParams":
        """Собрать параметры из словаря (секция `classification`)."""
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ===========================================================================
#  Вспомогательные вычисления
# ===========================================================================

def _clean(values: Sequence[float]) -> list[float]:
    """Убрать NaN/None из последовательности."""
    out: list[float] = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv):
            continue
        out.append(fv)
    return out


def _mean(values: Sequence[float]) -> float:
    """Среднее арифметическое. Пустая последовательность → 0.0."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _compute_reference_points(
    values: Sequence[float], edge_frac: float
) -> tuple[float, float, float]:
    """
    Опорные точки интервала: (sp_top, sp_mid, sp_bot).
    Доли считаем от количества точек.
    """
    n = len(values)
    k = max(1, int(round(n * edge_frac)))

    sp_top = _mean(values[:k])
    sp_bot = _mean(values[-k:])

    center = n // 2
    start = max(0, center - k // 2)
    end = min(n, start + k)
    sp_mid = _mean(values[start:end])

    return sp_top, sp_mid, sp_bot


def _compute_frac_core(
    values: Sequence[float], sp_mid: float, core_threshold: float
) -> float:
    """
    Доля точек, близких к sp_mid (в пределах core_threshold).

    Различает trapezoid-middle / symmetric / v-shape:
        широкое плато → frac_core большая
        парабола      → средняя
        острый угол   → маленькая
    """
    n = len(values)
    if n == 0:
        return 0.0
    in_core = sum(1 for v in values if abs(v - sp_mid) < core_threshold)
    return in_core / n


def _has_multiple_minima(
    values: Sequence[float],
    cutoff: float,
    min_run_length: int = 5,
) -> bool:
    """
    Проверка переслаивания: два и более УСТОЙЧИВЫХ захода в песок,
    разделённых устойчивым глинистым участком.

    Устойчивый = длиной ≥ min_run_length точек.
    Это фильтрует шум — одиночные точки выше/ниже cutoff не считаются.

    Параметры
    ---------
    values : последовательность float
    cutoff : float — порог SP_norm (0.60)
    min_run_length : int — минимальная длина «полки» (5 точек)

    Возвращает True, если найдено ≥ 2 устойчивых песчаных «полки».
    """
    n = len(values)
    if n < 2 * min_run_length + 1:
        return False

    # Сжимаем в последовательность меток: True — песок, False — глина.
    is_sand = [v < cutoff for v in values]

    # Ищем непрерывные «полки» песка длиной ≥ min_run_length.
    sand_runs = 0
    i = 0
    while i < n:
        if is_sand[i]:
            # Начало песчаной полки — считаем длину.
            j = i
            while j < n and is_sand[j]:
                j += 1
            if j - i >= min_run_length:
                sand_runs += 1
            i = j
        else:
            i += 1

    return sand_runs >= 2


# ===========================================================================
#  Ядро: классификация одного интервала
# ===========================================================================

def classify_form(
    sp_smooth: Sequence[float],
    params: Optional[ClassificationParams] = None,
    config: Optional[dict] = None,
) -> dict:
    """
    Определить форму одного коллекторного интервала.

    Возвращает dict с полями:
        form_type         — тип формы
        confidence        — 0.0..1.0
        reason            — короткое объяснение
        sp_top, sp_mid, sp_bot
        diff_top_bot, diff_top_mid, diff_bot_mid
        frac_core         — доля точек в ядре вокруг sp_mid
        n_points, rules_version
    """
    if params is None:
        params = ClassificationParams.from_dict(config)

    clean = _clean(sp_smooth)
    n = len(clean)

    result: dict = {
        "form_type": "uncertain",
        "confidence": 0.0,
        "reason": "",
        "sp_top": None, "sp_mid": None, "sp_bot": None,
        "diff_top_bot": None, "diff_top_mid": None, "diff_bot_mid": None,
        "frac_core": None,
        "n_points": n,
        "rules_version": params.rules_version,
    }

    # -------------------------------------------------------------------
    # Шаг 1. Мало данных
    # -------------------------------------------------------------------
    if n < params.min_points:
        result["reason"] = f"мало точек ({n} < {params.min_points})"
        return result

    # --- Опоры ---------------------------------------------------------
    sp_top, sp_mid, sp_bot = _compute_reference_points(clean, params.edge_frac)
    result["sp_top"] = sp_top
    result["sp_mid"] = sp_mid
    result["sp_bot"] = sp_bot

    diff_top_bot = sp_top - sp_bot
    diff_top_mid = sp_top - sp_mid
    diff_bot_mid = sp_bot - sp_mid

    result["diff_top_bot"] = diff_top_bot
    result["diff_top_mid"] = diff_top_mid
    result["diff_bot_mid"] = diff_bot_mid

    frac_core = _compute_frac_core(clean, sp_mid, params.core_threshold)
    result["frac_core"] = frac_core

    cutoff = params.reservoir_cutoff

        # -------------------------------------------------------------------
    # Шаг 2. M-форма (переслаивание)
    # -------------------------------------------------------------------
    # Признак 1: песчаные кровля и подошва, глинистый центр
    # (опоры сверху и снизу — в песчаной зоне, центр — в глинистой).
    # Признак 2: два и более УСТОЙЧИВЫХ захода в песок — переслаивание
    # без явного «глинистого центра» по опорам.
        # M-форма — переслаивание внутри интервала.
    # Ловим только по опорам: явная M-образная форма.
    # Тонкие переслаивания по всей длине кривой не ловим —
    # для этого нужны другие признаки, которых у нас пока нет
    # (см. docs/DECISIONS_LOG.md).
    is_m_shape = (
        sp_top < cutoff
        and sp_bot < cutoff
        and sp_mid > cutoff + params.slope_threshold
    )

    if is_m_shape:
        result["form_type"] = "uncertain"
        result["confidence"] = 0.30
        result["reason"] = ("возможно, переслаивание — "
                            "проверьте сегментацию")
        return result

        # -------------------------------------------------------------------
    # Шаг 3. Цилиндр: ровное плато на песчаном уровне
    # -------------------------------------------------------------------
    # Цилиндр = всё плато на песчаном уровне. Условие:
    #   • все три опоры в песчаной зоне;
    #   • верх и низ совпадают (|diff_top_bot| < threshold);
    #   • ядро вокруг sp_mid широкое (frac_core ≥ frac_core_trapezoid).
    #
    # Отличие от V-формы: у V-формы верх и низ тоже совпадают,
    # но ядро узкое (только вершина V).
    if (
        sp_top < cutoff
        and sp_mid < cutoff
        and sp_bot < cutoff
        and abs(diff_top_bot) < params.threshold_diff
        and frac_core >= params.frac_core_trapezoid
    ):
        result["form_type"] = "cylinder"
        result["confidence"] = min(1.0, frac_core * 1.2)
        result["reason"] = "ровное плато на песчаном уровне"
        return result

    # -------------------------------------------------------------------
    # Шаг 4. Глина — песок — глина
    # -------------------------------------------------------------------
    # sp_top и sp_bot примерно равны, sp_mid заметно ниже (песок в центре).
    top_bot_close = abs(diff_top_bot) < params.threshold_diff
    mid_is_low = (
        sp_mid < sp_top - params.slope_threshold
        and sp_mid < sp_bot - params.slope_threshold
    )

    if top_bot_close and mid_is_low:
        # Различаем по ширине ядра.
        if frac_core >= params.frac_core_trapezoid:
            result["form_type"] = "trapezoid-middle"
            result["confidence"] = min(1.0, 0.5 + frac_core)
            result["reason"] = "широкое плато песка в середине"
        elif frac_core >= params.frac_core_v_shape:
            result["form_type"] = "symmetric"
            result["confidence"] = min(1.0, 0.5 + frac_core)
            result["reason"] = "плавная чаша песка (парабола)"
        else:
            result["form_type"] = "v-shape"
            result["confidence"] = min(1.0, 0.5 + (params.frac_core_v_shape
                                                    - frac_core) * 3)
            result["reason"] = "острый угол, песчаный пик в центре"
        return result

    # -------------------------------------------------------------------
    # Шаг 5. Bell: глина сверху, песок снизу
    # -------------------------------------------------------------------
    if diff_top_bot > params.threshold_diff:
        conf = min(1.0, diff_top_bot / params.threshold_diff - 0.5)
        result["form_type"] = "bell"
        result["confidence"] = max(0.0, conf)
        result["reason"] = "глина сверху, песок снизу"
        return result

    # -------------------------------------------------------------------
    # Шаг 6. Funnel: песок сверху, глина снизу
    # -------------------------------------------------------------------
    if diff_top_bot < -params.threshold_diff:
        conf = min(1.0, abs(diff_top_bot) / params.threshold_diff - 0.5)
        result["form_type"] = "funnel"
        result["confidence"] = max(0.0, conf)
        result["reason"] = "песок сверху, глина снизу"
        return result

    # -------------------------------------------------------------------
    # Шаг 7. Ничего не подошло
    # -------------------------------------------------------------------
    result["form_type"] = "uncertain"
    result["confidence"] = 0.30
    result["reason"] = "ни одно правило не сработало"
    return result


# ===========================================================================
#  Обёртка: классификация всех интервалов скважины
# ===========================================================================

def classify_intervals(
    df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    params: ClassificationParams,
    smooth_window: int = 5,
) -> pd.DataFrame:
    """
    Прогнать классификацию по всем интервалам скважины.

    Возвращает копию intervals_df с новыми колонками:
        form_type, confidence, reason, rules_version,
        sp_top, sp_mid, sp_bot,
        diff_top_bot, diff_top_mid, diff_bot_mid,
        frac_core
    """
    new_cols = [
        "form_type", "confidence", "reason", "rules_version",
        "sp_top", "sp_mid", "sp_bot",
        "diff_top_bot", "diff_top_mid", "diff_bot_mid",
        "frac_core",
    ]

    if intervals_df.empty:
        out = intervals_df.copy()
        for col in new_cols:
            out[col] = pd.Series(dtype=object)
        return out

    depth = df["depth_md"].to_numpy(dtype=float)
    sp_norm = df["SP_norm"].to_numpy(dtype=float)
    sp_smooth_full = smooth(sp_norm, smooth_window)

    rows = []
    for _, iv in intervals_df.iterrows():
        rec = iv.to_dict()

        # Неколлекторные интервалы форму не имеют.
        if rec["kind"] != "reservoir":
            rec.update({
                "form_type": "non_reservoir",
                "confidence": 1.0,
                "reason": "неколлектор — форма не определяется",
                "rules_version": params.rules_version,
                "sp_top": None, "sp_mid": None, "sp_bot": None,
                "diff_top_bot": None, "diff_top_mid": None,
                "diff_bot_mid": None,
                "frac_core": None,
            })
            rows.append(rec)
            continue

        # Маска точек интервала по MD.
        mask = (depth >= rec["top_md"]) & (depth <= rec["bottom_md"])
        sp_slice = sp_smooth_full[mask]

        info = classify_form(sp_slice, params=params)
        rec.update(info)
        rows.append(rec)

    ordered_cols = list(intervals_df.columns) + new_cols
    result = pd.DataFrame(rows)[ordered_cols]
    return result


__all__ = [
    "RULES_VERSION",
    "ClassificationParams",
    "classify_form",
    "classify_intervals",
]
# -*- coding: utf-8 -*-
"""
Сводная типизация пласта (шаг F).

Одна строка на пласт-контейнер: собирает все признаки,
посчитанные на уровнях A (features) и B (intervals).

См. docs/FEATURES.md, §6; ADR-004 в docs/DECISIONS_LOG.md.

Ключевые колонки результата:
    thickness_reservoir_tvdss — суммарная мощность коллекторов, м
    ntg_tvdss                 — доля коллектора в мощности пласта
    n_reservoir               — количество коллекторных интервалов
    dominant_form             — доминирующая форма
    dominant_size             — доминирующий размер
    dominant_position         — доминирующее положение
    mean_confidence           — средняя confidence
    container_type            — короткая метка
    container_summary         — текстовое описание
    container_intervals_json  — JSON-массив интервалов

Правило «доминирования» (ADR-004):
    • основной критерий — суммарная мощность интервалов;
    • тай-брейк при близости (разница < dominance_margin × total):
      выше confidence; при равенстве — выше по глубине.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd


def _aggregate_by_key(
    intervals: list[dict],
    key: str,
) -> list[tuple[str, float, float]]:
    """
    Свернуть интервалы по значениям ключа (form / size / position).

    Возвращает список кортежей:
        (значение, сумма_мощностей, средняя_confidence)
    Отсортированный по убыванию суммы мощностей.
    """
    groups: dict[str, dict] = {}
    for iv in intervals:
        value = iv.get(key)
        if value is None:
            continue
        thickness = float(iv["thickness_tvdss"])
        confidence = float(iv.get("confidence", 0.0) or 0.0)

        if value not in groups:
            groups[value] = {"total": 0.0, "conf_sum": 0.0, "n": 0}
        groups[value]["total"] += thickness
        groups[value]["conf_sum"] += confidence
        groups[value]["n"] += 1

    result = []
    for value, g in groups.items():
        mean_conf = g["conf_sum"] / g["n"] if g["n"] > 0 else 0.0
        result.append((value, g["total"], mean_conf))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def _pick_dominant(
    aggregated: list[tuple[str, float, float]],
    total_reservoir: float,
    dominance_margin: float,
) -> Optional[str]:
    """
    Выбрать доминирующее значение.

    Правило (ADR-004):
        • если разница мощностей между топ-1 и топ-2 <
          dominance_margin × total_reservoir → тай-брейк по confidence;
        • при равенстве confidence — берём первое (оно выше по мощности,
          а при полной ничьей — выше по глубине, так как список уже
          отсортирован).

    Возвращает значение (строку) или None, если список пуст.
    """
    if not aggregated:
        return None
    if len(aggregated) == 1:
        return aggregated[0][0]

    top1_value, top1_total, top1_conf = aggregated[0]
    top2_value, top2_total, top2_conf = aggregated[1]

    if total_reservoir <= 0:
        return top1_value

    diff = top1_total - top2_total
    # Порог близости: 10% от суммарной мощности резервуаров.
    threshold = dominance_margin * total_reservoir

    if diff < threshold and top2_conf > top1_conf:
        # Близко по мощности, но у топ-2 выше confidence.
        return top2_value

    return top1_value


def _build_container_type(
    n_reservoir: int,
    dominant_form: Optional[str],
    dominant_size: Optional[str],
    dominant_position: Optional[str],
    ntg: float,
) -> str:
    """
    Короткая метка типа пласта.

    Формат:
        no-sand                           — NTG = 0, нет коллекторов
        single-{form}-{size}-{position}   — один интервал
        multi-{form}-{size}-{position}    — два и более
    """
    if ntg <= 0 or n_reservoir == 0:
        return "no-sand"

    prefix = "single" if n_reservoir == 1 else "multi"
    form = dominant_form or "uncertain"
    size = dominant_size or "unknown"
    position = dominant_position or "unknown"
    return f"{prefix}-{form}-{size}-{position}"


def _build_summary(intervals: list[dict]) -> str:
    """
    Текстовое описание пласта для человека.

    Примеры:
        "нет коллекторов"
        "1 пропласток: cylinder small top (6.57 м)"
        "2 пропластка: bell medium top (6.66 м); symmetric small bottom (1.09 м)"
    """
    n = len(intervals)
    if n == 0:
        return "нет коллекторов"

    parts = []
    for iv in intervals:
        form = iv.get("form_type", "?")
        size = iv.get("size_class", "?")
        pos = iv.get("position_in_container", "?")
        th = float(iv["thickness_tvdss"])
        parts.append(f"{form} {size} {pos} ({th:.2f} м)")

    # Склонение: 1 пропласток, 2-4 пропластка, 5+ пропластков.
    if n == 1:
        word = "пропласток"
    elif 2 <= n <= 4:
        word = "пропластка"
    else:
        word = "пропластков"

    return f"{n} {word}: " + "; ".join(parts)


def _build_intervals_json(intervals: list[dict]) -> str:
    """
    JSON-массив интервалов пласта.

    Каждый элемент:
        {"form", "size", "position", "thickness", "confidence",
         "top_tvdss", "bottom_tvdss"}
    """
    payload = []
    for iv in intervals:
        payload.append({
            "form": iv.get("form_type"),
            "size": iv.get("size_class"),
            "position": iv.get("position_in_container"),
            "thickness": float(iv["thickness_tvdss"]),
            "confidence": float(iv.get("confidence", 0.0) or 0.0),
            "top_tvdss": float(iv["top_tvdss"]),
            "bottom_tvdss": float(iv["bottom_tvdss"]),
        })
    return json.dumps(payload, ensure_ascii=False)


def compute_container_types(
    features_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    layers: list[dict],
    cfg: dict,
) -> pd.DataFrame:
    """
    Обогатить features_df сводными признаками уровня C.

    Параметры
    ---------
    features_df : pd.DataFrame
        Результат compute_container_features (одна строка на пласт).
    intervals_df : pd.DataFrame
        Результат classify_size_and_position. Нужны колонки:
        layer_name, kind, thickness_tvdss, form_type, size_class,
        position_in_container, confidence, top_tvdss, bottom_tvdss.
    layers : list[dict]
        Пласты-контейнеры из df.attrs["layers"].
    cfg : dict
        Секция container_type из config.yaml.

    Возвращает
    ----------
    pd.DataFrame — копия features_df + сводные колонки.
    """
    dominance_margin = cfg.get("dominance_margin", 0.10)

    new_cols = [
        "thickness_reservoir_tvdss", "ntg_tvdss", "n_reservoir",
        "dominant_form", "dominant_size", "dominant_position",
        "mean_confidence",
        "container_type", "container_summary", "container_intervals_json",
    ]

    if features_df.empty:
        out = features_df.copy()
        for col in new_cols:
            out[col] = pd.Series(dtype=object)
        return out

    # Индексируем интервалы по имени пласта.
    intervals_by_layer: dict[str, list[dict]] = {}
    for _, iv in intervals_df.iterrows():
        if iv["kind"] != "reservoir":
            continue
        intervals_by_layer.setdefault(iv["layer_name"], []).append(iv.to_dict())

    rows = []
    for _, row in features_df.iterrows():
        rec = row.to_dict()
        name = rec["layer_name"]
        layer_intervals = intervals_by_layer.get(name, [])

        # Мощность контейнера.
        thick_container = float(rec.get("thickness_tvdss") or 0.0)

        # Суммарная мощность коллекторов.
        total_reservoir = sum(
            float(iv["thickness_tvdss"]) for iv in layer_intervals
        )
        rec["thickness_reservoir_tvdss"] = total_reservoir
        rec["n_reservoir"] = len(layer_intervals)

        # NTG. Если мощность контейнера 0 — ntg = 0.
        if thick_container > 0:
            rec["ntg_tvdss"] = total_reservoir / thick_container
        else:
            rec["ntg_tvdss"] = 0.0

        # Если коллекторов нет — no-sand.
        if not layer_intervals:
            rec["dominant_form"] = None
            rec["dominant_size"] = None
            rec["dominant_position"] = None
            rec["mean_confidence"] = 0.0
            rec["container_type"] = "no-sand"
            rec["container_summary"] = "нет коллекторов"
            rec["container_intervals_json"] = "[]"
            rows.append(rec)
            continue

        # Доминирующие характеристики — по мощности, при близости — по confidence.
        agg_form = _aggregate_by_key(layer_intervals, "form_type")
        agg_size = _aggregate_by_key(layer_intervals, "size_class")
        agg_pos = _aggregate_by_key(layer_intervals, "position_in_container")

        rec["dominant_form"] = _pick_dominant(
            agg_form, total_reservoir, dominance_margin
        )
        rec["dominant_size"] = _pick_dominant(
            agg_size, total_reservoir, dominance_margin
        )
        rec["dominant_position"] = _pick_dominant(
            agg_pos, total_reservoir, dominance_margin
        )

        # Средняя confidence по всем интервалам пласта.
        confs = [float(iv.get("confidence", 0.0) or 0.0)
                 for iv in layer_intervals]
        rec["mean_confidence"] = sum(confs) / len(confs) if confs else 0.0

        # Итоговые строки.
        rec["container_type"] = _build_container_type(
            n_reservoir=len(layer_intervals),
            dominant_form=rec["dominant_form"],
            dominant_size=rec["dominant_size"],
            dominant_position=rec["dominant_position"],
            ntg=rec["ntg_tvdss"],
        )
        # Список интервалов сортируем сверху вниз по TVDSS кровли.
        sorted_ivs = sorted(
            layer_intervals,
            key=lambda x: -float(x["top_tvdss"]),
        )
        rec["container_summary"] = _build_summary(sorted_ivs)
        rec["container_intervals_json"] = _build_intervals_json(sorted_ivs)

        rows.append(rec)

    result = pd.DataFrame(rows)

    # Сохраняем порядок колонок: старые + новые.
    ordered = list(features_df.columns) + [
        c for c in new_cols if c not in features_df.columns
    ]
    return result[ordered]


__all__ = ["compute_container_types"]
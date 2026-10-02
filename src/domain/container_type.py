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
    form_in_label: Optional[str],
    dominant_size: Optional[str],
    dominant_position: Optional[str],
    ntg: float,
    container_state: str,
) -> str:
    """
    Короткая метка типа пласта (формат v3.0).

    Формат (docs/CLASSIFICATION_RULES_v3.md, §9.2):
        no-sand                            — NTG = 0, нет коллекторов
        {N}-interbedded-{size}-{position}  — переслаивание
        {N}-{form}-{size}-{position}       — все интервалы одной формы
        {N}-mixed-{size}-{position}        — интервалы разных форм

    где N — количество интервалов (1, 2, 3, ...).
    """
    if ntg <= 0 or n_reservoir == 0 or container_state == "no-sand":
        return "no-sand"

    size = dominant_size or "unknown"
    position = dominant_position or "unknown"

    # Переслаивание — специальная метка без формы.
    if container_state == "interbedded":
        return f"{n_reservoir}-interbedded-{size}-{position}"

    form = form_in_label or "unknown-shape"
    return f"{n_reservoir}-{form}-{size}-{position}"


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


def _compute_container_state(
    n_reservoir: int,
    layer_intervals: list[dict],
    cfg: dict,
) -> str:
    """
    Определяет состояние пласта (container_state) по v3.0 §9.1.

    Значения:
        no-sand        — 0 интервалов
        single-anomaly — 1 интервал
        multi-anomaly  — 2..interbedded_min_count-1 интервалов
        interbedded    — >= interbedded_min_count интервалов

    Примечание о пороге 5.
    Изначально v3.0 §9.1 предполагал ещё правило «>= половина
    микроинтервалов < 1 м → interbedded». Оно убрано, потому что:
      • сегментация фильтрует интервалы по MD, а мощность мы
        считаем в TVDSS. Граничные значения (0.99–1.00 м) дают
        ложные срабатывания;
      • после сегментации микроинтервалов практически не остаётся;
      • смысл «interbedded» — это много пропластков (5+), а не
        ситуация «2 пропластка, один из них тонкий».

    Параметр layer_intervals сейчас не используется, но оставлен
    в сигнатуре: возможно, позже вернёмся к учёту микроинтервалов
    (например, для пластов с 5+ интервалами, где это осмысленно).
    """
    if n_reservoir == 0:
        return "no-sand"
    if n_reservoir == 1:
        return "single-anomaly"

    interbedded_min = cfg.get("interbedded_min_count", 5)
    if n_reservoir >= interbedded_min:
        return "interbedded"
    return "multi-anomaly"

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
        "container_state",
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
            rec["container_state"] = "no-sand"
            rec["dominant_form"] = None
            rec["dominant_size"] = None
            rec["dominant_position"] = None
            rec["mean_confidence"] = 0.0
            rec["container_type"] = "no-sand"
            rec["container_summary"] = "нет коллекторов"
            rec["container_intervals_json"] = "[]"
            rows.append(rec)
            continue

        # Состояние пласта (single/multi/interbedded).
        rec["container_state"] = _compute_container_state(
            n_reservoir=len(layer_intervals),
            layer_intervals=layer_intervals,
            cfg=cfg,
        )

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

        # Определяем форму для метки: если все интервалы одной формы —
        # эта форма; если разные — "mixed".
        unique_forms = {
            iv.get("form_type") for iv in layer_intervals
            if iv.get("form_type") is not None
        }
        if len(unique_forms) == 1:
            form_in_label = next(iter(unique_forms))
        elif len(unique_forms) == 0:
            form_in_label = "unknown-shape"
        else:
            form_in_label = "mixed"

        # Итоговые строки.
        rec["container_type"] = _build_container_type(
            n_reservoir=len(layer_intervals),
            form_in_label=form_in_label,
            dominant_size=rec["dominant_size"],
            dominant_position=rec["dominant_position"],
            ntg=rec["ntg_tvdss"],
            container_state=rec["container_state"],
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
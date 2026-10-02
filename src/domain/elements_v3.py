# -*- coding: utf-8 -*-
"""
Разбиение коллекторного интервала на элементы (v3.0).

См. docs/CLASSIFICATION_RULES_v3.md, §3, §5, §6, §7.

Аномалия (reservoir-интервал из сегментации) описывается через
элементы:
    • Кровля (обязательный) — переход от неколлектора сверху
      к песчанику.
    • Плато (опциональный) — стабильный участок в средней части.
    • Подошва (обязательный) — переход от песчаника к
      неколлектору снизу.

Плато — интегральное понятие. Если внутри интервала несколько
плато-кандидатов одного класса (участки класса 2/3/4 длиной
≥ min_plateau_thickness_tvdss), они объединяются в одно
интегральное плато этого класса. Плато других классов не
поглощаются — они становятся частью кровли или подошвы.

Доминирующий класс плато — тот, у которого суммарная мощность
плато-кандидатов максимальна. Тай-брейк при равенстве — по
среднему aSP (выше = чище песок).

Модуль v3.0. Работает параллельно с classify_form.py (v2.x).
Пока используется только для пилота — интерфейс может меняться.
"""

from __future__ import annotations

import json

import numpy as np

from src.domain.segmentation import smooth


# Классы Муромцева, которые считаются «песком» — могут быть плато.
# Класс 0 (глина) и 1 (алевролит глинистый) — не плато.
_PLATEAU_CLASSES = (2, 3, 4)

# Сколько точек берём с края элемента, чтобы посчитать start_asp / end_asp.
# 5 точек = 0.5 м при шаге 0.1 м. Устойчиво к одиночному шуму.
_DEFAULT_EDGE_K = 5


# ===========================================================================
#  Вспомогательные функции
# ===========================================================================

def _candidate_to_dict(c):
    """Преобразует dict-кандидата в JSON-совместимый dict."""
    return {
        "class": int(c["class"]),
        "top_tvdss": float(c["top_tvdss"]),
        "bottom_tvdss": float(c["bottom_tvdss"]),
        "length_tvdss": float(c["length_tvdss"]),
        "mean_asp": (None if np.isnan(c["mean_asp"])
                     else float(c["mean_asp"])),
    }


def _find_plateau_candidates(asp_disc, asp, depth, min_length, min_asp):
    """
    Находит все плато-кандидаты — непрерывные участки одного класса
    из {2, 3, 4} длиной ≥ min_length в TVDSS и средним aSP ≥ min_asp.

    Параметры
    ---------
    asp_disc : np.ndarray[float]
        Дискретный лог (0..4 или NaN).
    asp : np.ndarray[float]
        Сырой aSP — для расчёта среднего значения на участке.
    depth : np.ndarray[float]
        Глубины в TVDSS, м.
    min_length : float
        Минимальная длина участка в TVDSS, м.
    min_asp : float
        Минимальный средний aSP участка. Кандидаты с меньшим
        средним отбрасываются — это, скорее, граница коллектора,
        чем песчаный уровень.

    Возвращает
    ----------
    list[dict] — кандидаты с полями:
        class, start_idx, end_idx, top_tvdss, bottom_tvdss,
        length_tvdss, mean_asp.
    """
    candidates = []
    n = len(asp_disc)
    if n == 0:
        return candidates

    i = 0
    while i < n:
        cls = asp_disc[i]

        # Пропускаем не-пластовые классы (0, 1) и NaN.
        if np.isnan(cls) or int(cls) not in _PLATEAU_CLASSES:
            i += 1
            continue

        # Нашли начало сегмента класса cls — считаем длину.
        j = i
        while j + 1 < n and asp_disc[j + 1] == cls:
            j += 1

        # Сегмент [i, j]. Считаем его геометрию.
        seg_depth = depth[i:j + 1]
        seg_asp = asp[i:j + 1]

        valid_d = seg_depth[~np.isnan(seg_depth)]
        valid_a = seg_asp[~np.isnan(seg_asp)]

        # Нужно хотя бы 2 точки, чтобы измерить длину.
        if len(valid_d) < 2:
            i = j + 1
            continue

        # TVDSS: глубже = более отрицательное. Верх = max, низ = min.
        top_tvdss = float(np.max(valid_d))
        bottom_tvdss = float(np.min(valid_d))
        length = top_tvdss - bottom_tvdss

        if length < min_length:
            i = j + 1
            continue

        mean_asp = float(np.mean(valid_a)) if len(valid_a) > 0 else np.nan

        # Фильтр по среднему aSP: плато с недостаточно высоким
        # средним не считаем плато. Это отсекает участки класса 2
        # на границе коллектора (asp ~0.45-0.50).
        if np.isnan(mean_asp) or mean_asp < min_asp:
            i = j + 1
            continue

        candidates.append({
            "class": int(cls),
            "start_idx": i,
            "end_idx": j,
            "top_tvdss": top_tvdss,
            "bottom_tvdss": bottom_tvdss,
            "length_tvdss": length,
            "mean_asp": mean_asp,
        })

        i = j + 1

    return candidates


def _weighted_mean_asp(candidates_of_class):
    """
    Средневзвешенное aSP по плато-кандидатам одного класса.
    Вес — длина в TVDSS. Если данных нет — 0.0.
    """
    total_w = 0.0
    weighted_sum = 0.0
    for c in candidates_of_class:
        if not np.isnan(c["mean_asp"]):
            weighted_sum += c["mean_asp"] * c["length_tvdss"]
            total_w += c["length_tvdss"]
    if total_w <= 0:
        return 0.0
    return weighted_sum / total_w


def _pick_dominant_class(candidates):
    """
    Возвращает доминирующий класс плато.

    Правило: по суммарной мощности плато-кандидатов. Тай-брейк при
    равенстве — по среднему aSP (выше = доминирует).

    Возвращает int (2 / 3 / 4) или None, если кандидатов нет.
    """
    if not candidates:
        return None

    # Суммарная мощность по классам.
    total_by_class = {}
    by_class = {}
    for c in candidates:
        cls = c["class"]
        total_by_class[cls] = total_by_class.get(cls, 0.0) + c["length_tvdss"]
        by_class.setdefault(cls, []).append(c)

    max_len = max(total_by_class.values())
    top_classes = [cls for cls, l in total_by_class.items() if l == max_len]

    if len(top_classes) == 1:
        return top_classes[0]

    # Тай-брейк — по среднему aSP.
    best_cls = None
    best_asp = -1.0
    for cls in top_classes:
        wm = _weighted_mean_asp(by_class[cls])
        if wm > best_asp:
            best_asp = wm
            best_cls = cls

    return best_cls if best_cls is not None else top_classes[0]


def _build_integral_plateau(candidates, dominant_class):
    """
    Строит интегральное плато из кандидатов одного класса.

    Возвращает dict:
        class, top_idx, bottom_idx, top_tvdss, bottom_tvdss,
        length_tvdss, mean_asp, segments (list[dict]).
    Или None, если кандидатов нужного класса нет.
    """
    segs = [c for c in candidates if c["class"] == dominant_class]
    if not segs:
        return None

    segs = sorted(segs, key=lambda x: x["start_idx"])
    first = segs[0]
    last = segs[-1]

    top_tvdss = first["top_tvdss"]
    bottom_tvdss = last["bottom_tvdss"]
    length = top_tvdss - bottom_tvdss
    mean_asp = _weighted_mean_asp(segs)

    return {
        "class": dominant_class,
        "top_idx": first["start_idx"],
        "bottom_idx": last["end_idx"],
        "top_tvdss": top_tvdss,
        "bottom_tvdss": bottom_tvdss,
        "length_tvdss": length,
        "mean_asp": mean_asp,
        "segments": segs,
    }


def _classify_boundary_type(length_tvdss, thickness, boundary_frac,
                            boundary_max_tvdss):
    """
    М (мгновенный) или Н (наклонный).

    Граница мгновенная, если её длина ≤ min(boundary_frac × thickness,
    boundary_max_tvdss). Иначе — наклонная.
    """
    threshold = min(boundary_frac * thickness, boundary_max_tvdss)
    if length_tvdss <= threshold:
        return "М"
    return "Н"


def _compute_modifier(asp_values, smooth_window, zigzag_threshold,
                      zigzag_fraction):
    """
    Модификатор элемента: 'пологий' / 'зубчатый' / 'abs'.

    Элемент зубчатый, если доля точек, отклоняющихся от локального
    среднего больше чем на zigzag_threshold, превышает zigzag_fraction.

    'abs' — если в элементе нет точек (пустой сегмент).
    """
    n = len(asp_values)
    if n == 0:
        return "abs"

    valid = ~np.isnan(asp_values)
    if valid.sum() < 3:
        return "пологий"

    smoothed = smooth(asp_values, smooth_window)
    both = valid & ~np.isnan(smoothed)
    if not both.any():
        return "пологий"

    deviations = np.abs(asp_values[both] - smoothed[both])
    fraction = float(np.mean(deviations > zigzag_threshold))

    if fraction > zigzag_fraction:
        return "зубчатый"
    return "пологий"


def _compute_plateau_trend(asp_values, threshold):
    """
    Тренд плато: 'flat' / 'rising' / 'falling' / 'symmetric'.

    Считается по сырому aSP внутри плато. Плато делится на три
    примерно равные части (голова / середина / хвост), считаются
    средние, и по их разностям определяется характер.
    """
    valid = asp_values[~np.isnan(asp_values)]
    n = len(valid)
    if n < 6:
        return "flat"

    k = n // 3
    head = float(np.mean(valid[:k]))
    mid = float(np.mean(valid[k:2 * k]))
    tail = float(np.mean(valid[2 * k:]))

    # Симметричное: середина заметно выше (или ниже) обоих краёв.
    mid_high = (mid > head + threshold) and (mid > tail + threshold)
    mid_low = (mid < head - threshold) and (mid < tail - threshold)
    if mid_high or mid_low:
        return "symmetric"

    diff_ht = tail - head
    if diff_ht > threshold:
        return "rising"
    if diff_ht < -threshold:
        return "falling"
    return "flat"


def _edge_mean(arr, start, end, from_start, k):
    """
    Среднее по k валидным точкам с одного края сегмента [start, end).
    from_start=True — с начала; False — с конца. None если данных нет.
    """
    seg = arr[start:end]
    valid = seg[~np.isnan(seg)]
    if len(valid) == 0:
        return None
    if from_start:
        return float(np.mean(valid[:k]))
    return float(np.mean(valid[-k:]))


def _mean_gradient(asp, depth, start, end):
    """
    Средний градиент aSP по элементу (aSP/м).

    ΔaSP / Δz, где z — абсолютная глубина (положительная вниз).
    Поскольку TVDSS отрицательно вниз, Δz = −ΔTVDSS.

    Возвращает float или None (если точек мало).
    """
    seg_a = asp[start:end]
    seg_d = depth[start:end]
    valid = ~np.isnan(seg_a) & ~np.isnan(seg_d)
    if valid.sum() < 2:
        return None

    a = seg_a[valid]
    d = seg_d[valid]
    delta_a = float(a[-1] - a[0])
    delta_z = -float(d[-1] - d[0])

    if abs(delta_z) < 1e-9:
        return None
    return delta_a / delta_z


def _empty_result():
    """Заготовка результата на случай, если данных недостаточно."""
    return {
        # Структурные
        "n_elements": 2,
        "has_plateau": False,
        "plateau_position": "absent",
        "max_position": None,

        # Кровля
        "top_type": "abs",
        "top_length_m": 0.0,
        "top_length_frac": 0.0,
        "top_start_asp": None,
        "top_end_asp": None,
        "top_mean_gradient": None,
        "top_modifier": "abs",

        # Плато
        "plateau_top_tvdss": None,
        "plateau_bottom_tvdss": None,
        "plateau_length_m": None,
        "plateau_length_frac": None,
        "plateau_dominant_class": None,
        "plateau_mean_asp": None,
        "plateau_trend": "absent",
        "plateau_modifier": "abs",

        # Подошва
        "bot_type": "abs",
        "bot_length_m": 0.0,
        "bot_length_frac": 0.0,
        "bot_start_asp": None,
        "bot_end_asp": None,
        "bot_mean_gradient": None,
        "bot_modifier": "abs",

        # Детали плато
        "n_plateaus": 0,
        "plateau_segments_json": "[]",
    }


# ===========================================================================
#  Главная функция
# ===========================================================================

def split_interval_into_elements(
    asp_slice,
    depth_slice,
    asp_disc_slice,
    params,
    smooth_window,
):
    """
    Разбивает один коллекторный интервал на элементы (v3.0).

    Параметры
    ---------
    asp_slice : np.ndarray[float]
        Сырой aSP для точек интервала, сверху вниз.
    depth_slice : np.ndarray[float]
        TVDSS тех же точек (глубже = более отрицательное).
    asp_disc_slice : np.ndarray[float]
        Дискретный лог (0..4 или NaN) для тех же точек.
    params : dict
        Секция classification_v3 из config.yaml.
    smooth_window : int
        Окно сглаживания (из preprocessing).

    Возвращает
    ----------
    dict — полное описание элементов (см. _empty_result для ключей).
    """
    result = _empty_result()
    n = len(asp_slice)

    # --- Слишком короткий интервал ---
    if n < 3:
        return result

    # --- Границы интервала в TVDSS ---
    interval_top_tvdss = float(depth_slice[0])
    interval_bottom_tvdss = float(depth_slice[-1])
    thickness = interval_top_tvdss - interval_bottom_tvdss

    if thickness <= 0:
        return result

    # --- Сглаженный aSP (для max_position) ---
    asp_smoothed = smooth(asp_slice, smooth_window)

    # --- Ищем плато-кандидаты ---
    candidates = _find_plateau_candidates(
        asp_disc_slice, asp_slice, depth_slice,
        params["min_plateau_thickness_tvdss"],
        params["min_plateau_asp"],
    )
    result["n_plateaus"] = len(candidates)
    result["plateau_segments_json"] = json.dumps(
        [_candidate_to_dict(c) for c in candidates],
        ensure_ascii=False,
    )

    # --- Определяем интегральное плато ---
    plateau = None
    if candidates:
        dominant_class = _pick_dominant_class(candidates)
        plateau = _build_integral_plateau(candidates, dominant_class)

    # --- Разбиение на элементы ---
    if plateau is not None:
        result["has_plateau"] = True
        result["n_elements"] = 3

        p_top_idx = plateau["top_idx"]
        p_bot_idx = plateau["bottom_idx"]

        # Кровля: индексы [0, p_top_idx) — до начала плато.
        top_idx_start, top_idx_end = 0, p_top_idx
        # Плато: [p_top_idx, p_bot_idx] включительно.
        plateau_slice = asp_slice[p_top_idx:p_bot_idx + 1]
        # Подошва: (p_bot_idx, n).
        bot_idx_start, bot_idx_end = p_bot_idx + 1, n

        top_length = interval_top_tvdss - plateau["top_tvdss"]
        bot_length = plateau["bottom_tvdss"] - interval_bottom_tvdss

        # Позиция плато внутри аномалии.
        plateau_center = (plateau["top_tvdss"] + plateau["bottom_tvdss"]) / 2.0
        rel = (interval_top_tvdss - plateau_center) / thickness
        if rel < 0.33:
            result["plateau_position"] = "top"
        elif rel > 0.67:
            result["plateau_position"] = "bottom"
        else:
            result["plateau_position"] = "middle"

        result["plateau_top_tvdss"] = plateau["top_tvdss"]
        result["plateau_bottom_tvdss"] = plateau["bottom_tvdss"]
        result["plateau_length_m"] = plateau["length_tvdss"]
        result["plateau_length_frac"] = plateau["length_tvdss"] / thickness
        result["plateau_dominant_class"] = plateau["class"]
        result["plateau_mean_asp"] = plateau["mean_asp"]
        result["plateau_trend"] = _compute_plateau_trend(
            plateau_slice, params["plateau_trend_threshold"]
        )
        result["plateau_modifier"] = _compute_modifier(
            plateau_slice, smooth_window,
            params["zigzag_threshold"], params["zigzag_fraction"],
        )

    else:
        # Треугольник: плато нет, разбиваем по максимуму aSP.
        result["has_plateau"] = False
        result["n_elements"] = 2

        valid = ~np.isnan(asp_smoothed)
        if not valid.any():
            return result

        asp_for_max = np.where(valid, asp_smoothed, -np.inf)
        max_idx = int(np.argmax(asp_for_max))

        max_tvdss = float(depth_slice[max_idx])
        max_position = (interval_top_tvdss - max_tvdss) / thickness
        result["max_position"] = max_position

        top_length = interval_top_tvdss - max_tvdss
        bot_length = max_tvdss - interval_bottom_tvdss

        top_idx_start, top_idx_end = 0, max_idx
        bot_idx_start, bot_idx_end = max_idx, n

    # --- Общие поля: длины и типы границ ---
    result["top_length_m"] = top_length
    result["top_length_frac"] = top_length / thickness
    result["bot_length_m"] = bot_length
    result["bot_length_frac"] = bot_length / thickness

    result["top_type"] = _classify_boundary_type(
        top_length, thickness,
        params["boundary_frac"], params["boundary_max_tvdss"],
    )
    result["bot_type"] = _classify_boundary_type(
        bot_length, thickness,
        params["boundary_frac"], params["boundary_max_tvdss"],
    )

    # --- Модификаторы границ ---
    top_seg = asp_slice[top_idx_start:top_idx_end]
    bot_seg = asp_slice[bot_idx_start:bot_idx_end]

    result["top_modifier"] = _compute_modifier(
        top_seg, smooth_window,
        params["zigzag_threshold"], params["zigzag_fraction"],
    )
    result["bot_modifier"] = _compute_modifier(
        bot_seg, smooth_window,
        params["zigzag_threshold"], params["zigzag_fraction"],
    )

    # --- start/end aSP границ ---
    result["top_start_asp"] = _edge_mean(
        asp_slice, top_idx_start, top_idx_end,
        from_start=True, k=_DEFAULT_EDGE_K,
    )
    result["top_end_asp"] = _edge_mean(
        asp_slice, top_idx_start, top_idx_end,
        from_start=False, k=_DEFAULT_EDGE_K,
    )
    result["bot_start_asp"] = _edge_mean(
        asp_slice, bot_idx_start, bot_idx_end,
        from_start=True, k=_DEFAULT_EDGE_K,
    )
    result["bot_end_asp"] = _edge_mean(
        asp_slice, bot_idx_start, bot_idx_end,
        from_start=False, k=_DEFAULT_EDGE_K,
    )

    # --- Градиенты ---
    result["top_mean_gradient"] = _mean_gradient(
        asp_slice, depth_slice, top_idx_start, top_idx_end,
    )
    result["bot_mean_gradient"] = _mean_gradient(
        asp_slice, depth_slice, bot_idx_start, bot_idx_end,
    )

    return result


__all__ = ["split_interval_into_elements"]
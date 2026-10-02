"""
Дискретный лог aSP_disc по 5 классам Муромцева.

Зачем:
    Классификация формы v3.0 (см. docs/CLASSIFICATION_RULES_v3.md)
    использует дискретный лог для:
      • быстрого выделения аномалий (класс >= 2);
      • поиска кандидатов в плато (длинные участки одного класса);
      • уточнения границ элементов.

    Дискретный лог — вспомогательный. Точная форма (типы границ,
    тренды, модификаторы) считается по сырому aSP.

Классы (границы (lower, upper], нижняя не включается):
    0: aSP <= 0.2          — глина
    1: 0.2 < aSP <= 0.4    — алевролит глинистый
    2: 0.4 < aSP <= 0.6    — алевролит / песчаник
    3: 0.6 < aSP <= 0.8    — песчаник среднезернистый
    4: aSP > 0.8           — песчаник крупнозернистый

Короткие участки:
    Участок одного класса короче min_class_thickness_tvdss
    (стартово 0.5 м) присоединяется к соседу с большей мощностью
    в TVDSS. При равной мощности — к соседу с большим числом
    точек. Если сосед один — к нему. Если оба соседа NaN или
    отсутствуют — участок не трогаем.

NaN:
    Точки с aSP = NaN (вне рабочего интервала, пропуски) дают
    aSP_disc = NaN. NaN-точки не сливаются ни с чем и служат
    естественной границей участков.
"""

import numpy as np


# Верхние границы классов Муромцева по aSP.
# Правило интервалов: (lower, upper] — нижняя не входит, верхняя входит.
_CLASS_UPPER_BOUNDS = (0.2, 0.4, 0.6, 0.8)


def _classify_asp_values(asp):
    """
    Присваивает каждой точке aSP класс Муромцева (0..4) или NaN.

    Параметры
    ---------
    asp : np.ndarray[float]

    Возвращает
    ----------
    np.ndarray[float] — классы 0.0 .. 4.0 или NaN.
                        float, чтобы можно было хранить NaN.
    """
    asp = np.asarray(asp, dtype=float)
    disc = np.full(len(asp), np.nan, dtype=float)

    valid = ~np.isnan(asp)
    a = asp[valid]

    # Каскад np.where: сначала самая узкая граница (<= 0.2),
    # затем более широкие. Порядок важен, чтобы классы не пересекались.
    class_values = np.where(
        a <= _CLASS_UPPER_BOUNDS[0], 0.0,
        np.where(
            a <= _CLASS_UPPER_BOUNDS[1], 1.0,
            np.where(
                a <= _CLASS_UPPER_BOUNDS[2], 2.0,
                np.where(
                    a <= _CLASS_UPPER_BOUNDS[3], 3.0,
                    4.0,
                ),
            ),
        ),
    )

    disc[valid] = class_values
    return disc


def _segment_thickness(depth_slice):
    """
    Мощность сегмента в TVDSS: |последняя - первая| валидная точка.

    Нужна, чтобы сравнивать соседей по правилу v3.0:
    «присоединяем к соседу с большей мощностью».
    """
    valid = depth_slice[~np.isnan(depth_slice)]
    if valid.size < 2:
        return 0.0
    return float(abs(valid[-1] - valid[0]))


def _make_segment(start, end, cls, depth):
    """Собирает словарь-описание одного сегмента."""
    depth_slice = depth[start:end + 1]
    return {
        "start": start,
        "end": end,
        "cls": cls,
        "thickness": _segment_thickness(depth_slice),
        "n_points": end - start + 1,
    }


def _find_segments(disc, depth):
    """
    Разбивает массив классов на непрерывные сегменты.

    Сегмент — максимальная последовательность соседних точек
    с одинаковым классом. NaN-точки образуют отдельные сегменты
    (cls = NaN); они никогда не присоединяются.

    Возвращает
    ----------
    list[dict] — сегменты с полями:
        index      — порядковый номер (0..n-1);
        start, end — индексы в массиве (end включительно);
        cls        — значение класса (float или NaN);
        thickness  — мощность в TVDSS;
        n_points   — число точек.
    """
    n = len(disc)
    if n == 0:
        return []

    def _both_nan(a, b):
        return (isinstance(a, float) and np.isnan(a)
                and isinstance(b, float) and np.isnan(b))

    segments = []
    start = 0
    current = disc[0]

    for i in range(1, n):
        # Один сегмент, если классы равны. NaN == NaN в numpy даёт
        # False, поэтому проверяем особый случай отдельно.
        same = (disc[i] == current) or _both_nan(disc[i], current)
        if not same:
            segments.append(_make_segment(start, i - 1, current, depth))
            start = i
            current = disc[i]

    segments.append(_make_segment(start, n - 1, current, depth))

    for i, seg in enumerate(segments):
        seg["index"] = i

    return segments


def _is_nan_segment(seg):
    """True, если сегмент — это последовательность NaN-точек."""
    cls = seg["cls"]
    return isinstance(cls, float) and np.isnan(cls)


def _pick_neighbor_to_join(segments, seg_index):
    """
    Выбирает, к какому соседу присоединить короткий сегмент.

    Правило v3.0:
      • из двух валидных соседей (не NaN) — тот, у кого больше
        мощность в TVDSS;
      • при равной мощности — тот, у кого больше точек;
      • если сосед один — берём его;
      • если соседей нет или оба NaN — возвращаем None.

    Возвращает индекс соседа в списке segments или None.
    """
    left = segments[seg_index - 1] if seg_index > 0 else None
    right = (
        segments[seg_index + 1]
        if seg_index + 1 < len(segments)
        else None
    )

    candidates = []
    if left is not None and not _is_nan_segment(left):
        candidates.append(left)
    if right is not None and not _is_nan_segment(right):
        candidates.append(right)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]["index"]

    a, b = candidates
    if a["thickness"] > b["thickness"]:
        return a["index"]
    if b["thickness"] > a["thickness"]:
        return b["index"]
    # Равные мощности — по числу точек.
    if a["n_points"] >= b["n_points"]:
        return a["index"]
    return b["index"]


def _merge_short_segments(disc, depth, min_thickness):
    """
    Присоединяет короткие сегменты одного класса к соседям.

    Итеративно: находим короткий сегмент (< min_thickness по TVDSS),
    присоединяем его к соседу, пересчитываем сегменты. Повторяем,
    пока такие сегменты есть.

    Защита от зацикливания: если сегмент не к чему присоединить
    (оба соседа NaN или отсутствуют) — пропускаем его. Если за
    полный проход ни одного присоединения не случилось — выходим.
    """
    disc = disc.copy()
    if len(disc) == 0:
        return disc

    while True:
        segments = _find_segments(disc, depth)
        merged_any = False

        for seg in segments:
            if _is_nan_segment(seg):
                continue
            if seg["thickness"] >= min_thickness:
                continue

            target_idx = _pick_neighbor_to_join(segments, seg["index"])
            if target_idx is None:
                continue

            target_cls = segments[target_idx]["cls"]
            disc[seg["start"]:seg["end"] + 1] = target_cls
            merged_any = True
            break  # сегменты изменились — пересчитываем с начала

        if not merged_any:
            break

    return disc


def compute_asp_disc(df, min_class_thickness_tvdss):
    """
    Считает колонку aSP_disc (5 классов Муромцева) для таблицы скважины.

    Требует в df колонки:
      • aSP          — нормированный ПС (0 = глина, 1 = песок);
      • depth_tvdss  — абсолютные отметки, м (для склейки коротких
                       сегментов по мощности).

    Параметры
    ---------
    df : pd.DataFrame
        Таблица из build_dataset (или аналогичная).
    min_class_thickness_tvdss : float
        Минимальная мощность участка одного класса в TVDSS, м.
        Всё короче — присоединяется к соседу.
        Значение берётся из config["classification_v3"].

    Возвращает
    ----------
    np.ndarray[float]
        Значения aSP_disc: 0.0 .. 4.0 или NaN.
        NaN там же, где NaN в aSP (вне рабочего интервала).
    """
    asp = df["aSP"].to_numpy(dtype=float)
    depth = df["depth_tvdss"].to_numpy(dtype=float)

    disc = _classify_asp_values(asp)
    disc = _merge_short_segments(disc, depth, min_class_thickness_tvdss)

    return disc
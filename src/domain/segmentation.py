"""
Сегментация пласта-контейнера на интервалы коллектор/неколлектор.

См. docs/FEATURES.md, раздел 5.

Алгоритм:
    1. Сгладить SP_norm скользящим средним (окно из config).
    2. Применить порог SP_norm < threshold → reservoir / non_reservoir.
    3. Собрать непрерывные интервалы с одинаковой меткой.
    4. Объединить интервалы < min_thickness_m с соседями (по большинству).

Правила:
    - Минимальная мощность интервала — min_thickness_m (по умолчанию 1 м).
    - Все границы интервалов пересчитываются в TVDSS через интерполяцию
      по .dev. Мощность считаем в абсолютных отметках.
    - Если пласт полностью глинистый или полностью песчаный — возвращаем
      один интервал соответствующего типа.
"""
import pandas as pd  
import numpy as np
from dataclasses import dataclass



@dataclass
class Interval:
    """Один непрерывный интервал внутри пласта."""
    kind: str             # "reservoir" | "non_reservoir"
    top_md: float         # кровля, м MD
    bottom_md: float      # подошва, м MD
    top_tvdss: float      # кровля в АО, м
    bottom_tvdss: float   # подошва в АО, м
    thickness_md: float   # мощность по стволу, м
    thickness_tvdss: float  # мощность в АО, м  ← главная
    n_points: int         # сколько точек каротажа внутри


def smooth(values, window):
    """
    Скользящее среднее с сохранением NaN.

    Параметры:
        values : np.ndarray — массив значений (с NaN)
        window : int        — окно в точках (например, 5)

    Возвращает:
        np.ndarray той же длины.

    Правило:
        - NaN остаются NaN в результате (не заменяем на число).
        - Усредняем только валидные значения внутри окна.
        - На краях окно сужается (нет выходов за пределы массива).
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=float)
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = values[lo:hi]
        valid = window_vals[~np.isnan(window_vals)]
        if valid.size > 0:
            result[i] = valid.mean()

    return result


def _label_points(sp_smoothed, threshold):
    """
    Присваивает каждой точке метку reservoir/non_reservoir.

    Параметры:
        sp_smoothed : np.ndarray — сглаженная SP_norm (NaN, где данных нет)
        threshold   : float      — порог SP_norm (обычно 0.4)

    Возвращает:
        np.ndarray[object] — метки: "reservoir", "non_reservoir" или None.
        None там, где sp_smoothed = NaN.
    """
    labels = np.full(len(sp_smoothed), None, dtype=object)
    valid = ~np.isnan(sp_smoothed)
    labels[valid & (sp_smoothed < threshold)] = "reservoir"
    labels[valid & (sp_smoothed >= threshold)] = "non_reservoir"
    return labels


def _build_raw_intervals(depth_md, labels):
    """
    Собирает непрерывные интервалы по меткам.

    Возвращает list[dict] с полями:
        kind       — "reservoir" / "non_reservoir"
        top_md     — кровля, м MD
        bottom_md  — подошва, м MD (включительно с последней точкой)
        n_points   — число точек
        idx_start  — индекс первой точки в массиве depth_md
        idx_end    — индекс последней точки (включительно)

    Работает так:
        Идём по точкам, помним текущую метку и начало интервала.
        Когда метка меняется (или доходим до конца) — закрываем интервал.
    """
    intervals = []
    n = len(depth_md)

    if n == 0:
        return intervals

    current_kind = labels[0]
    start_idx = 0

    for i in range(1, n):
        if labels[i] != current_kind:
            # Закрываем предыдущий интервал
            if current_kind is not None:
                intervals.append({
                    "kind": current_kind,
                    "top_md": float(depth_md[start_idx]),
                    "bottom_md": float(depth_md[i - 1]),
                    "n_points": i - start_idx,
                    "idx_start": start_idx,
                    "idx_end": i - 1,
                })
            current_kind = labels[i]
            start_idx = i

    # Закрываем последний интервал
    if current_kind is not None:
        intervals.append({
            "kind": current_kind,
            "top_md": float(depth_md[start_idx]),
            "bottom_md": float(depth_md[n - 1]),
            "n_points": n - start_idx,
            "idx_start": start_idx,
            "idx_end": n - 1,
        })

    return intervals


def _merge_small_intervals(intervals, min_points):
    """
    Объединяет интервалы мощностью < min_points с соседями.

    Параметры:
        intervals  : list[dict] — из _build_raw_intervals
        min_points : int — минимальное число точек для сохранения интервала

    Возвращает:
        list[dict] — после слияния.

    Алгоритм:
        Идём по списку слева направо. Если интервал короткий —
        присоединяем его к соседу с большим числом точек
        (слева или справа). Границы нового соседа расширяются.

        Если короткий интервал оказался первым или последним
        (нет обоих соседей) — присоединяем к единственному.

        Крайний случай: пласт целиком короче min_points —
        возвращаем как есть, тип остаётся исходный. Это пласт
        попадёт в отдельную категорию "micro", но пока не трогаем.
    """
    if not intervals:
        return intervals

    # Специальный случай: единственный интервал
    if len(intervals) == 1:
        return intervals

    # Работаем копией, чтобы не портить оригинал
    result = [dict(iv) for iv in intervals]

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result):
            iv = result[i]
            if iv["n_points"] >= min_points:
                i += 1
                continue

            # Короткий интервал. Определяем, к кому присоединиться.
            has_left = i > 0
            has_right = i < len(result) - 1

            if has_left and has_right:
                # Кто больше — к тому и присоединяемся
                if result[i - 1]["n_points"] >= result[i + 1]["n_points"]:
                    target = i - 1
                else:
                    target = i + 1
            elif has_left:
                target = i - 1
            elif has_right:
                target = i + 1
            else:
                # Единственный интервал — но это случай len==1,
                # он обрабатывается выше
                i += 1
                continue

            # Расширяем целевой интервал до границ короткого
            if target < i:
                # Целевой слева
                result[target]["bottom_md"] = iv["bottom_md"]
                result[target]["idx_end"] = iv["idx_end"]
                result[target]["n_points"] += iv["n_points"]
            else:
                # Целевой справа
                result[target]["top_md"] = iv["top_md"]
                result[target]["idx_start"] = iv["idx_start"]
                result[target]["n_points"] += iv["n_points"]

            # Удаляем короткий интервал
            del result[i]
            changed = True
            # Не увеличиваем i — после удаления на этой позиции новый
            # интервал, нужно его перепроверить

    return result

def _merge_adjacent_same_type(intervals):
    """
    Объединяет соседние интервалы одного типа.

    Зачем: при сглаживании возможна одиночная точка, «переключившая»
    метку. После _merge_small_intervals она может остаться как
    отдельный короткий интервал, который при этом уже прошёл порог
    минимальной мощности. Результат — два подряд идущих интервала
    одного типа.

    Параметры:
        intervals : list[dict] — из _merge_small_intervals

    Возвращает:
        list[dict] — без соседних дубликатов по типу.
    """
    if len(intervals) < 2:
        return intervals

    result = [dict(intervals[0])]

    for iv in intervals[1:]:
        last = result[-1]
        if last["kind"] == iv["kind"]:
            # Тот же тип — расширяем последний до границ нового
            last["bottom_md"] = iv["bottom_md"]
            last["idx_end"] = iv["idx_end"]
            last["n_points"] += iv["n_points"]
        else:
            result.append(dict(iv))

    return result

def segment_layer(
    depth_md,
    sp_norm,
    top_tvdss_container,
    bottom_tvdss_container,
    md_to_tvdss_fn,
    cutoff=0.6,
    smooth_window=5,
    min_thickness_tvdss=1.0,
):
    """
    Сегментирует один пласт-контейнер.

    Параметры:
        depth_md              : np.ndarray — глубины по стволу, м MD
        sp_norm               : np.ndarray — SP_norm (0..1), NaN там, где нет
        top_tvdss_container   : float — кровля пласта в АО
        bottom_tvdss_container: float — подошва пласта в АО
        md_to_tvdss_fn        : callable — функция перевода MD → TVDSS
        threshold             : float — порог SP_norm для коллектора
        smooth_window         : int — окно сглаживания в точках
        min_thickness_tvdss   : float — минимальная мощность интервала в АО, м

    Возвращает:
        list[Interval] — сегменты пласта, отсортированные сверху вниз.
        Пустой список — если в пласте нет валидных точек SP_norm.
    """
    # --- 1. Сглаживание ---------------------------------------------
    sp_smoothed = smooth(sp_norm, smooth_window)

    # --- 2. Метки ---------------------------------------------------
    labels = _label_points(sp_smoothed, cutoff)

    # --- 3. Сырые интервалы -----------------------------------------
    raw = _build_raw_intervals(depth_md, labels)
    if not raw:
        return []

    # --- 4. Минимальный размер в точках -----------------------------
    # Оцениваем шаг каротажа: медиана разницы между соседними MD
    if len(depth_md) >= 2:
        step = float(np.median(np.diff(depth_md)))
    else:
        step = 0.1  # fallback

    # min_points: сколько точек нужно, чтобы мощность была >= min_thickness
    # Простая оценка: min_thickness / step. Так как мы работаем в MD,
    # а min_thickness задан в АО, для наклонных стволов MD-мощность
    # немного больше АО. Берём с запасом снизу — то есть
    # min_points = ceil(min_thickness / step). Строго по АО проверим позже.
    min_points = max(1, int(np.ceil(min_thickness_tvdss / step)))

     # --- 5. Слияние мелких ------------------------------------------
    merged = _merge_small_intervals(raw, min_points)

    # --- 5b. Слияние соседних одного типа ---------------------------
    merged = _merge_adjacent_same_type(merged)

    # --- 6. Обогащаем TVDSS и мощностями ----------------------------
    result = []
    for iv in merged:
        top_tvdss = md_to_tvdss_fn(iv["top_md"])
        bottom_tvdss = md_to_tvdss_fn(iv["bottom_md"])

        # Ограничиваем интервал границами контейнера
        top_tvdss = min(top_tvdss, top_tvdss_container)
        bottom_tvdss = max(bottom_tvdss, bottom_tvdss_container)

        thickness_tvdss = top_tvdss - bottom_tvdss  # оба отрицательные вниз
        thickness_md = iv["bottom_md"] - iv["top_md"]

        result.append(Interval(
            kind=iv["kind"],
            top_md=iv["top_md"],
            bottom_md=iv["bottom_md"],
            top_tvdss=top_tvdss,
            bottom_tvdss=bottom_tvdss,
            thickness_md=thickness_md,
            thickness_tvdss=thickness_tvdss,
            n_points=iv["n_points"],
        ))

    return result


def segment_all_layers(df, layers, md_to_tvdss_fn, cfg):
    """
    Прогоняет сегментацию по всем пластам скважины.

    Параметры:
        df              : pd.DataFrame из build_dataset
        layers          : list[dict] из df.attrs["layers"]
        md_to_tvdss_fn  : callable — MD → TVDSS
        cfg             : dict — секция segmentation из config.yaml

    Возвращает:
        pd.DataFrame — одна строка на интервал:
            layer_name, interval_index, kind, top_md, bottom_md,
            top_tvdss, bottom_tvdss, thickness_md, thickness_tvdss, n_points
    """
    cutoff = cfg.get("cutoff", 0.6)
    smooth_window = cfg.get("smooth_window", 5)
    min_thickness_tvdss = cfg.get("min_thickness_tvdss", 1.0)

    rows = []

    for lay in layers:
        name = lay["name"]
        mask = df["layer_name"] == name
        df_lay = df.loc[mask]

        if df_lay.empty:
            continue

        depth_md = df_lay["depth_md"].to_numpy(dtype=float)
        sp_norm = df_lay["SP_norm"].to_numpy(dtype=float)

        intervals = segment_layer(
            depth_md=depth_md,
            sp_norm=sp_norm,
            top_tvdss_container=lay["top_tvdss"],
            bottom_tvdss_container=lay["bottom_tvdss"],
            md_to_tvdss_fn=md_to_tvdss_fn,
            cutoff=cutoff,
            smooth_window=smooth_window,
            min_thickness_tvdss=min_thickness_tvdss,
        )

        for idx, iv in enumerate(intervals, start=1):
            rows.append({
                "layer_name":       name,
                "interval_index":   idx,
                "kind":             iv.kind,
                "top_md":           iv.top_md,
                "bottom_md":        iv.bottom_md,
                "top_tvdss":        iv.top_tvdss,
                "bottom_tvdss":     iv.bottom_tvdss,
                "thickness_md":     iv.thickness_md,
                "thickness_tvdss":  iv.thickness_tvdss,
                "n_points":         iv.n_points,
            })

    result = pd.DataFrame(rows)
    return result



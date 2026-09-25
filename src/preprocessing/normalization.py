"""
Нормализация кривых ГИС (приведение к безразмерной шкале 0..1).

Зачем:
    Разные скважины / разные операторы дают SP и GK в разных единицах
    и с разным уровнем. Нормализация убирает эту зависимость.

Подход:
    Опорные линии ищутся как перцентили распределения значений
    по всей скважине. Настраивается через config.yaml.

    Для SP и GK направления разные:
      SP: глина = высокое, песок = низкое  → (x - sand) / (shale - sand)
      GK: глина = высокое, песок = низкое  → (x - sand) / (shale - sand)
    Обе формулы дают 0 = песчаник, 1 = глина. Единая семантика.

Что делать с NaN:
    NaN остаются NaN. np.nanpercentile их игнорирует.

Возвращает:
    Тот же DataFrame с добавленными колонками SP_norm, GK_norm.
"""

import numpy as np
import pandas as pd


def _baseline_from_percentiles(values, shale_pct, sand_pct):
    """
    Считает опорные линии 'глина' и 'песок' по перцентилям.

    Параметры:
        values    : array-like — массив значений (с NaN)
        shale_pct : float — перцентиль для глины (обычно 95)
        sand_pct  : float — перцентиль для песка (обычно 5)

    Возвращает:
        (shale_value, sand_value) — оба float.

    Если валидных значений нет — возвращает (nan, nan).
    """
    arr = np.asarray(values, dtype=float)

    # Отбрасываем NaN, чтобы перцентили считались по реальным данным
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.nan, np.nan

    shale_value = float(np.percentile(valid, shale_pct))
    sand_value = float(np.percentile(valid, sand_pct))
    return shale_value, sand_value


def _normalize_curve(values, shale_value, sand_value):
    """
    Линейно нормирует кривую к [0, 1].

    Формула: (x - sand) / (shale - sand)
        0 → песчаник
        1 → глина

    Параметры:
        values      : array-like
        shale_value : float — опорное значение 'глина' (верх)
        sand_value  : float — опорное значение 'песок' (низ)

    Возвращает:
        np.ndarray, значения вне [0, 1] обрезаны до 0 и 1.

    Про деление на ноль:
        Если shale == sand, знаменатель ноль. Возвращаем NaN —
        пусть валидатор потом отдельно пожалуется.
    """
    arr = np.asarray(values, dtype=float)

    denom = shale_value - sand_value
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        # Нечего нормировать — кривая константа или пустая
        return np.full_like(arr, np.nan)

    norm = (arr - sand_value) / denom

    # Обрезаем хвосты за пределами [0, 1] — там шум/выбросы
    norm = np.clip(norm, 0.0, 1.0)

    return norm


def normalize_curves(df, cfg, interval=None):
    """
    Нормализует все кривые в DataFrame по настройкам из config.yaml.

    Параметры:
        df       : pd.DataFrame — таблица LAS (колонки: DEPTH, SP, GK)
        cfg      : dict — секция normalization из config.yaml
        interval : (top_md, bottom_md) или None
                   Если задан — нормализация (включая расчёт перцентилей)
                   выполняется ТОЛЬКО по точкам внутри интервала.
                   Значения вне интервала становятся NaN.

    Возвращает:
        Тот же DataFrame с колонками SP_norm, GK_norm.
        attrs["baselines"]     — (shale, sand) по каждой кривой
        attrs["interval"]      — применённый интервал (top_md, bottom_md)

    Логика:
        1. Строим булеву маску in_interval по глубинам DEPTH.
        2. Для каждой кривой:
           а) берём значения ТОЛЬКО внутри интервала
           б) по ним считаем перцентили (опорные линии)
           в) нормируем ВСЕ значения (полный массив)
           г) результат ВНЕ интервала → NaN
    """
    result = df.copy()
    baselines = {}

    # Глубины из LAS. Имя колонки — "DEPTH" (см. depth_mnemonic в config).
    depth = result["DEPTH"].to_numpy(dtype=float)

    # --- Маска точек внутри интервала -------------------------------
    if interval is not None:
        top_md, bottom_md = interval
        # Левая граница включительно, правая тоже — чтобы захватить
        # саму кровлю и последнюю точку забоя
        in_interval = (depth >= top_md) & (depth <= bottom_md)
    else:
        # Интервал не задан — берём всю скважину
        in_interval = np.ones(len(depth), dtype=bool)

    # --- Обработка каждой кривой ------------------------------------
    for canonical in ["SP", "GK"]:
        # Соответствующая секция в config: "sp" или "gk"
        curve_cfg_key = canonical.lower()
        if curve_cfg_key not in cfg:
            # На всякий случай — если в конфиге нет настроек для этой кривой
            baselines[canonical] = (np.nan, np.nan)
            continue

        if canonical not in result.columns:
            # Кривой нет в данных
            baselines[canonical] = (np.nan, np.nan)
            continue

        values = result[canonical].to_numpy(dtype=float)

        # а) значения только внутри интервала
        values_in_interval = values[in_interval]

        # б) перцентили по этим значениям
        shale, sand = _baseline_from_percentiles(
            values_in_interval,
            shale_pct=cfg[curve_cfg_key]["shale_percentile"],
            sand_pct=cfg[curve_cfg_key]["sand_percentile"],
        )

        # в) нормируем весь массив
        norm_full = _normalize_curve(values, shale, sand)

        # г) обнуляем вне интервала
        norm_full = np.asarray(norm_full, dtype=float)
        norm_full[~in_interval] = np.nan

        result[f"{canonical}_norm"] = norm_full
        baselines[canonical] = (shale, sand)

    # --- Атрибуты для отчёта ----------------------------------------
    result.attrs["baselines"] = baselines
    result.attrs["normalization_interval"] = interval

    return result
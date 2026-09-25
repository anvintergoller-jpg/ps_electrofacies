"""
Сборка единого набора данных по одной скважине.

Вход:
    - DataFrame LAS  (DEPTH, SP, GK)              — из las_reader
    - DataFrame .dev (MD, X, Y, Z, TVD, ...)      — из deviation_reader
    - DataFrame отбивок (marker_name, md, tvdss)  — из markers_reader

Выход:
    DataFrame с колонками:
        depth_md      — глубина по стволу (из LAS), м
        depth_tvd     — истинная вертикальная глубина, м
        depth_tvdss   — абсолютная отметка (TVD − KB), м
        SP, GK        — сырые кривые
        SP_norm, GK_norm — нормализованные (0..1)
        layer_name    — имя пласта или None, если вне пластов

    Плюс attrs:
        kb          — Kelly Bushing из .dev
        layers      — list[dict] с полными границами пластов
                      (включая последний — до конца данных)
        source_well — имя скважины

Правила:
    - Разметка точек пластов — по MD (depth_md).
    - Мощности (thickness_tvdss) — в абсолютных отметках.
    - Для последнего пласта подошва = max(MD из .dev), по TVDSS —
      Z последней точки .dev.
"""

import numpy as np
import pandas as pd

from src.ingestion.deviation_reader import md_to_tvd


def _mark_layers(depth_md, layers):
    """
    Для каждой глубины в depth_md определяет, в каком пласте она лежит.

    Параметры:
        depth_md : np.ndarray — массив глубин (MD) из LAS
        layers   : list[dict] — пласты из markers_to_layers
                                (у последнего bottom_md уже подставлен)

    Возвращает:
        np.ndarray[object] — имена пластов или None.
    """
    result = np.full(len(depth_md), None, dtype=object)

    for lay in layers:
        top = lay["top_md"]
        bottom = lay["bottom_md"]

        # Левая граница включительно, правая — исключительно,
        # чтобы граничные точки не попадали одновременно в два пласта
        mask = (depth_md >= top) & (depth_md < bottom)
        result[mask] = lay["name"]

    return result


def build_dataset(df_las, df_dev, df_markers, cfg, mnem_cfg=None):
    """
    Собирает единую таблицу по скважине.

    Параметры:
        df_las      : pd.DataFrame — из las_reader.read_las
        df_dev      : pd.DataFrame — из deviation_reader.read_deviation
        df_markers  : pd.DataFrame — из markers_reader.read_markers
        cfg         : dict         — полный config.yaml
        mnem_cfg    : не используется (для совместимости)

    Возвращает:
        pd.DataFrame — единая таблица.
    """
    # Импорт внутри функции — чтобы избежать циклов на уровне модулей
    from src.ingestion.markers_reader import markers_to_layers

    # --- 1. Каркас: копируем LAS ------------------------------------
    df = df_las.copy()

    # --- 2. Глубина по стволу ---------------------------------------
    depth_md = df["DEPTH"].to_numpy(dtype=float)

    # --- 3. TVD через интерполяцию по .dev --------------------------
    tvd = md_to_tvd(df_dev, depth_md)

    # --- 4. TVDSS = TVD − KB ----------------------------------------
    kb = df_dev.attrs.get("kb")
    if kb is None:
        # Если KB не нашли в шапке — считаем 0, но это подозрительно.
        # Валидатор отдельно предупредит.
        kb = 0.0

        # TVDSS в Petrel-стиле: отрицательное вниз, KB − TVD.
    # Совпадает с PVD auto из отбивок и столбцом Z из .dev.
    tvdss = kb - tvd

    df["depth_md"] = depth_md
    df["depth_tvd"] = tvd
    df["depth_tvdss"] = tvdss

    # DEPTH больше не нужен — есть depth_md
    df = df.drop(columns=["DEPTH"])

    # --- 5. Список пластов из отбивок -------------------------------
    # markers_to_layers возвращает список с полями:
    #   name, top_md, bottom_md, thickness_md,
    #   top_tvdss, bottom_tvdss, thickness_tvdss.
    # У последнего пласта bottom_md и bottom_tvdss = None.
    layers = markers_to_layers(df_markers)

    # --- 6. Доопределяем последний пласт ----------------------------
    # Подошва последнего пласта — конец данных в .dev
    md_max_dev = float(df_dev["MD"].max())

    # TVDSS последней точки .dev. В файле есть столбец Z = TVDSS.
    if "Z" in df_dev.columns:
        tvdss_max = float(df_dev["Z"].iloc[-1])
    else:
        # Резервный вариант: TVD − KB
        kb_local = df_dev.attrs.get("kb", 0.0)
        tvdss_max = float(df_dev["TVD"].iloc[-1]) - kb_local

    for lay in layers:
        if lay["bottom_md"] is None:
            lay["bottom_md"] = md_max_dev
            lay["bottom_tvdss"] = tvdss_max
            lay["thickness_md"] = md_max_dev - lay["top_md"]
            if lay["top_tvdss"] is not None:
                lay["thickness_tvdss"] = lay["top_tvdss"] - tvdss_max

    # --- 7. Разметка точек пластами по MD ---------------------------
    df["layer_name"] = _mark_layers(depth_md, layers)

    # --- 8. Атрибуты результата -------------------------------------
    df.attrs["kb"] = kb
    df.attrs["layers"] = layers
    df.attrs["source_well"] = cfg["pilot_well"]["name"]

    return df
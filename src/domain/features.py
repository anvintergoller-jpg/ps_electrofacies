"""
Расчёт признаков пласта-контейнера (уровень A).

См. docs/FEATURES.md, раздел 4.

Пока — только признаки уровня A (по пласту целиком).
Признаки уровня B (по каждому коллекторному интервалу) —
после реализации сегментации (шаг C).
"""

import numpy as np
import pandas as pd
from src.domain.segmentation import smooth


def layer_stats(df_layer):
    """
    Считает простую статистику по точкам одного пласта.

    Параметры:
        df_layer : pd.DataFrame — строки пласта (с колонкой SP_norm)

    Возвращает dict:
        n_points     — всего точек
        n_sp_valid   — точек с валидным SP_norm
        sp_coverage  — доля валидных, 0..1
    """
    n = len(df_layer)
    if n == 0:
        return {"n_points": 0, "n_sp_valid": 0, "sp_coverage": 0.0}

    if "SP_norm" not in df_layer.columns:
        return {"n_points": n, "n_sp_valid": 0, "sp_coverage": 0.0}

    sp = df_layer["SP_norm"].to_numpy(dtype=float)
    n_valid = int(np.sum(~np.isnan(sp)))

    return {
        "n_points": n,
        "n_sp_valid": n_valid,
        "sp_coverage": n_valid / n if n > 0 else 0.0,
    }

def compute_lithology_profile(sp_norm_values, smooth_window=5):
    """
    Считает доли точек в каждом из пяти классов αПС.

    Классы (по Муромцеву):
        pct_shale       — глина             αПС 0.0-0.2  SP_norm 0.8-1.0
        pct_silty_shale — алевролит глин.   αПС 0.2-0.4  SP_norm 0.6-0.8
        pct_silt        — алевролиты/пески  αПС 0.4-0.6  SP_norm 0.4-0.6
        pct_sand_m      — песчаник ср/з     αПС 0.6-0.8  SP_norm 0.2-0.4
        pct_sand_c      — песчаник кр/з     αПС 0.8-1.0  SP_norm 0.0-0.2

    Сглаживание применяется для устойчивости: одиночные точки
    на границе корзин дают шум, сглаживание его убирает.

    Параметры:
        sp_norm_values : np.ndarray — SP_norm пласта (с NaN)
        smooth_window  : int — окно сглаживания, из config

    Возвращает:
        dict с пятью ключами pct_*, сумма = 1.0 (или 0.0, если нет данных).
    """
    arr = np.asarray(sp_norm_values, dtype=float)
    smoothed = smooth(arr, smooth_window)

    valid = smoothed[~np.isnan(smoothed)]
    n = valid.size

    if n == 0:
        return {
            "pct_shale": 0.0,
            "pct_silty_shale": 0.0,
            "pct_silt": 0.0,
            "pct_sand_m": 0.0,
            "pct_sand_c": 0.0,
        }

    return {
        "pct_shale":       float(np.sum(valid >= 0.8) / n),
        "pct_silty_shale": float(np.sum((valid >= 0.6) & (valid < 0.8)) / n),
        "pct_silt":        float(np.sum((valid >= 0.4) & (valid < 0.6)) / n),
        "pct_sand_m":      float(np.sum((valid >= 0.2) & (valid < 0.4)) / n),
        "pct_sand_c":      float(np.sum(valid < 0.2) / n),
    }

def compute_container_features(df, layers):
    """
    Считает признаки уровня A для всех пластов скважины.

    Параметры:
        df     : pd.DataFrame из build_dataset
                 (содержит depth_md, SP_norm, layer_name)
        layers : list[dict] из df.attrs["layers"]
                 (в каждом — top_md, bottom_md, top_tvdss, ...)

    Возвращает:
        pd.DataFrame, одна строка на пласт.
    """
    rows = []

    for lay in layers:
        name = lay["name"]

        # Отбираем точки пласта по MD
        mask = df["layer_name"] == name
        df_lay = df.loc[mask]

        stats = layer_stats(df_lay)

        # Литологический профиль — по SP_norm пласта
        if "SP_norm" in df_lay.columns:
            sp = df_lay["SP_norm"].to_numpy(dtype=float)
        else:
            sp = np.array([])
        litho = compute_lithology_profile(sp, smooth_window=5)

        rows.append({
            "layer_name":      name,
            "top_md":          lay.get("top_md"),
            "bottom_md":       lay.get("bottom_md"),
            "thickness_md":    lay.get("thickness_md"),
            "top_tvdss":       lay.get("top_tvdss"),
            "bottom_tvdss":    lay.get("bottom_tvdss"),
            "thickness_tvdss": lay.get("thickness_tvdss"),
            "n_points":        stats["n_points"],
            "n_sp_valid":      stats["n_sp_valid"],
            "sp_coverage":     stats["sp_coverage"],
            "pct_shale":       litho["pct_shale"],
            "pct_silty_shale": litho["pct_silty_shale"],
            "pct_silt":        litho["pct_silt"],
            "pct_sand_m":      litho["pct_sand_m"],
            "pct_sand_c":      litho["pct_sand_c"],
        })

    result = pd.DataFrame(rows)

    # Сортировка по кровле (сверху вниз — по убыванию TVDSS)
    if not result.empty and "top_tvdss" in result.columns:
        result = result.sort_values(
            "top_tvdss", ascending=False
        ).reset_index(drop=True)

    return result
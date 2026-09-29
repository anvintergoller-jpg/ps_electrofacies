# -*- coding: utf-8 -*-
"""
Сопоставление пластов между скважинами (шаг H).

Читает сводную таблицу summary_all_wells_<ts>.csv и строит
набор артефактов для анализа одноимённых пластов в разных
скважинах.

Артефакты (в data/processed/):
    cross_well_matrix.csv        — матрица «пласт × скважина»,
                                    в ячейках container_type;
    cross_well_matrix_<metric>.csv — та же матрица, но с числами
                                    (по умолчанию NTG);
    cross_well_counts.csv        — счётчики по (пласт, тип):
                                    сколько скважин каждого типа;
    cross_well_atypical.csv      — строки, где container_type
                                    встречается только в 1 скважине
                                    (нетипичные пласты).

Плюс текстовый отчёт в консоль: матрица + счётчики +
нетипичные + агрегаты по числовым признакам.

См. docs/FEATURES.md, §8.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_container_matrix(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Матрица «пласт × скважина» со значениями container_type.

    Строки — layer_name (все пласты, встречающиеся хотя бы
    в одной скважине).
    Столбцы — well_name (все скважины из summary_df).
    Ячейка — строка container_type, либо NaN (если пласта в
    этой скважине нет).
    """
    return summary_df.pivot_table(
        index="layer_name",
        columns="well_name",
        values="container_type",
        aggfunc="first",
    )


def build_numeric_matrix(
    summary_df: pd.DataFrame, metric: str
) -> pd.DataFrame:
    """
    Матрица «пласт × скважина» с числовой метрикой.

    Та же форма, что и матрица container_type.
    """
    if metric not in summary_df.columns:
        raise ValueError(
            f"Метрика {metric!r} отсутствует в summary_df. "
            f"Доступные: {list(summary_df.columns)}"
        )
    matrix = summary_df.pivot_table(
        index="layer_name",
        columns="well_name",
        values=metric,
        aggfunc="first",
    )
    return matrix.round(3)


def build_counts(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Счётчики по (layer_name, container_type).

    Возвращает DataFrame:
        layer_name, container_type, n_wells, wells
    Отсортированный: сначала по layer_name (по кровле — атрибут
    слоя не нужен, сортируем по алфавиту), внутри — по убыванию
    n_wells.
    """
    rows = []
    grouped = summary_df.groupby(["layer_name", "container_type"])
    for (layer, ctype), grp in grouped:
        rows.append({
            "layer_name": layer,
            "container_type": ctype,
            "n_wells": len(grp),
            "wells": "; ".join(sorted(grp["well_name"].tolist())),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["layer_name", "n_wells", "container_type"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return df


def build_atypical(
    summary_df: pd.DataFrame, counts_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Нетипичные пласты: (layer, well), где container_type
    встречается только в 1 скважине.

    Возвращает DataFrame с ключевыми числовыми признаками —
    чтобы геолог сразу видел контекст.
    """
    # Пары (layer, type), у которых n_wells == 1.
    singles = counts_df.loc[
        counts_df["n_wells"] == 1, ["layer_name", "container_type"]
    ]

    # Соединяем с summary по этим парам.
    atypical = summary_df.merge(
        singles, on=["layer_name", "container_type"], how="inner",
    )

    cols = [
        "layer_name", "well_name", "container_type",
        "ntg_tvdss", "thickness_tvdss", "thickness_reservoir_tvdss",
        "n_reservoir", "dominant_form", "dominant_size",
        "dominant_position", "mean_confidence",
    ]
    cols = [c for c in cols if c in atypical.columns]
    atypical = atypical[cols].reset_index(drop=True)
    return atypical.sort_values(["layer_name", "well_name"])


def _print_matrix(matrix: pd.DataFrame, title: str):
    """Печатает матрицу в консоль с выравниванием по колонкам."""
    print(f"\n  {title}")
    print("  " + "-" * 80)
    if matrix.empty:
        print("  (пусто)")
        return

    # Заголовок.
    well_cols = [str(c) for c in matrix.columns]
    header = f"  {'Пласт':<14}" + "".join(f"{w:<20}" for w in well_cols)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for layer, row in matrix.iterrows():
        cells = []
        for w in matrix.columns:
            v = row[w]
            if pd.isna(v):
                cells.append(f"{'—':<20}")
            else:
                cells.append(f"{str(v):<20}")
        print(f"  {str(layer):<14}" + "".join(cells))


def build_cross_well_summary(
    summary_df: pd.DataFrame,
    output_dir: Path,
    cfg: dict,
) -> dict:
    """
    Построить все артефакты шага H.

    Параметры
    ---------
    summary_df : pd.DataFrame
        Сводная таблица (одна строка на пласт × скважину).
    output_dir : Path
        Куда сохранять артефакты (обычно data/processed).
    cfg : dict
        Секция cross_well_summary из config.yaml.

    Возвращает
    ----------
    dict с путями сохранённых файлов.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric = cfg.get("numeric_metric", "ntg_tvdss")

    # --- Артефакты ---------------------------------------------------
    matrix = build_container_matrix(summary_df)
    numeric_matrix = build_numeric_matrix(summary_df, metric)
    counts_df = build_counts(summary_df)
    atypical_df = build_atypical(summary_df, counts_df)

    # --- Сохранение файлов ------------------------------------------
    matrix_path = output_dir / "cross_well_matrix.csv"
    numeric_path = output_dir / f"cross_well_matrix_{metric}.csv"
    counts_path = output_dir / "cross_well_counts.csv"
    atypical_path = output_dir / "cross_well_atypical.csv"

    matrix.to_csv(matrix_path, encoding="utf-8-sig")
    numeric_matrix.to_csv(numeric_path, encoding="utf-8-sig")
    counts_df.to_csv(counts_path, index=False, encoding="utf-8-sig")
    atypical_df.to_csv(atypical_path, index=False, encoding="utf-8-sig")

    # --- Отчёт в консоль --------------------------------------------
    print(f"\n  Скважин: {summary_df['well_name'].nunique()}")
    print(f"  Уникальных пластов: {summary_df['layer_name'].nunique()}")
    print(f"  Строк (пласт × скважина): {len(summary_df)}")

    _print_matrix(matrix, "МАТРИЦА «ПЛАСТ × СКВАЖИНА» (container_type):")
    _print_matrix(numeric_matrix, f"МАТРИЦА ПО {metric}:")

    # --- Счётчики ----------------------------------------------------
    print(f"\n  СЧЁТЧИКИ ПО ПЛАСТАМ:")
    print("  " + "-" * 80)
    for layer in counts_df["layer_name"].unique():
        sub = counts_df[counts_df["layer_name"] == layer]
        n_wells_layer = int(sub["n_wells"].sum())
        n_types = len(sub)
        print(f"\n  {layer}  ({n_wells_layer} скважин, "
              f"{n_types} {'тип' if n_types == 1 else 'типов'})")
        for _, r in sub.iterrows():
            marker = " ⚠ уникальный" if r["n_wells"] == 1 else ""
            print(f"    {r['n_wells']}× {r['container_type']:<32}"
                  f" {r['wells']}{marker}")

    # --- Нетипичные --------------------------------------------------
    print(f"\n  НЕТИПИЧНЫЕ ПЛАСТЫ (уникальный тип в выборке):")
    print("  " + "-" * 80)
    if atypical_df.empty:
        print("  (нет — все типы встречаются минимум в 2 скважинах)")
    else:
        for _, r in atypical_df.iterrows():
            ntg = r.get("ntg_tvdss", None)
            ntg_str = f"NTG={ntg:.3f}" if pd.notna(ntg) else "NTG=—"
            print(f"    {r['layer_name']:<14} {r['well_name']:<12} "
                  f"{r['container_type']:<32} {ntg_str}")

    # --- Агрегаты числовых признаков ---------------------------------
    print(f"\n  ЧИСЛОВАЯ СВОДКА ПО ПЛАСТАМ:")
    print("  " + "-" * 80)
    header = (f"  {'Пласт':<14} {'N wells':>8} "
              f"{'NTG mean':>10} {'NTG min':>9} {'NTG max':>9} "
              f"{'Tres mean':>11}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for layer in sorted(summary_df["layer_name"].unique()):
        sub = summary_df[summary_df["layer_name"] == layer]
        n = len(sub)
        ntg = sub["ntg_tvdss"].dropna()
        tres = sub["thickness_reservoir_tvdss"].dropna()
        ntg_mean = ntg.mean() if not ntg.empty else 0.0
        ntg_min = ntg.min() if not ntg.empty else 0.0
        ntg_max = ntg.max() if not ntg.empty else 0.0
        tres_mean = tres.mean() if not tres.empty else 0.0
        print(f"  {layer:<14} {n:>8d} "
              f"{ntg_mean:>10.3f} {ntg_min:>9.3f} {ntg_max:>9.3f} "
              f"{tres_mean:>11.2f}")

    return {
        "matrix": matrix_path,
        "numeric_matrix": numeric_path,
        "counts": counts_path,
        "atypical": atypical_path,
    }


__all__ = ["build_cross_well_summary"]
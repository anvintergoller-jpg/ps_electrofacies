"""
Чтение отбивок (WellTops) из Excel-выгрузки Petrel.

Формат:
    - Один общий файл, внутри — строки по всем скважинам.
    - Имя скважины в столбце "Well identifier (Well name)".
    - Имя маркера (горизонта) в столбце "Surface".
    - MD — measured depth, в столбце "MD".
    - TVDSS — в столбце "PVD auto".
    - Прочих столбцов ~40, нам нужны только эти 4.

Возвращает:
    pd.DataFrame с колонками: marker_name, md, tvdss.
    Строки отсортированы по MD по возрастанию.

Дополнительно:
    Функция markers_to_layers превращает список кровель
    в список пластов (top, bottom). Последний пласт — до конца.
"""

import numpy as np
import pandas as pd


def read_markers(path, well_name, config):
    """
    Читает отбивки для одной скважины.

    Параметры:
        path      : str/Path — Excel-файл
        well_name : str — имя скважины (например, "227_2263")
        config    : dict — секция markers из config.yaml

    Возвращает:
        pd.DataFrame с колонками:
            marker_name : str
            md          : float (Measured Depth, м)
            tvdss       : float (Depth Subsea, м)
    """
    # Имена столбцов — из конфига
    well_col = config["well_column_name"]
    md_col = config["md_column_name"]
    name_col = config["marker_name_column_name"]
    tvdss_col = config["tvdss_column_name"]

    # pd.read_excel определяет движок сам. Для .xlsx нужен openpyxl —
    # он устанавливается как зависимость pandas.
    df = pd.read_excel(path)

    # --- Нормализация имён столбцов -------------------------------
    # Petrel иногда добавляет пробелы в конце имён столбцов.
    # Приводим к чистому виду, чтобы сравнивать без сюрпризов.
    df.columns = [str(c).strip() for c in df.columns]

    # --- Проверка наличия нужных столбцов ------------------------
    required = [well_col, md_col, name_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"В файле {path} нет столбцов: {missing}.\n"
            f"Доступны: {list(df.columns)}"
        )

    # --- Фильтр по скважине --------------------------------------
    # .astype(str).str.strip() — приводим к строке и убираем пробелы,
    # чтобы "227_2263" и " 227_2263 " совпали
    df = df[df[well_col].astype(str).str.strip() == well_name].copy()

    if df.empty:
        raise ValueError(
            f"В файле {path} нет строк для скважины '{well_name}'"
        )

    # --- Собираем результат ---------------------------------------
    result = pd.DataFrame({
        "marker_name": df[name_col].astype(str).str.strip(),
        # pd.to_numeric с errors="coerce" — если ячейка вдруг текст,
        # превратит в NaN вместо падения
        "md": pd.to_numeric(df[md_col], errors="coerce"),
    })

    # TVDSS — необязательный столбец, но полезный
    if tvdss_col in df.columns:
        result["tvdss"] = pd.to_numeric(df[tvdss_col], errors="coerce")
    else:
        result["tvdss"] = np.nan

    # --- Сортировка по MD ----------------------------------------
    # Petrel не гарантирует порядок строк. Сортируем сами,
    # чтобы пласты шли сверху вниз.
    result = result.sort_values("md").reset_index(drop=True)

    # --- Фильтр по списку маркеров (если задан) -------------------
    markers_to_use = config.get("markers_to_use", "all")
    if markers_to_use != "all":
        result = result[result["marker_name"].isin(markers_to_use)]
        result = result.reset_index(drop=True)

    # --- Убираем строки без MD -----------------------------------
    result = result.dropna(subset=["md"]).reset_index(drop=True)

    return result


def markers_to_layers(markers_df):
    """
    Преобразует список кровель в список пластов.

    Возвращает list[dict], каждый dict содержит:
        name             — имя пласта (по кровле)
        top_md           — кровля в MD, м
        bottom_md        — подошва в MD (= кровля следующего), м
        thickness_md     — мощность по стволу, м
        top_tvdss        — кровля в абсолютных отметках, м
        bottom_tvdss     — подошва в абсолютных отметках, м
        thickness_tvdss  — мощность в АО, м  ← главная метрика

    Для последнего пласта bottom_md = None и bottom_tvdss = None.
    Они заполняются позже — в build_dataset, по данным из .dev.
    """
    layers = []
    n = len(markers_df)

    for i in range(n):
        row = markers_df.iloc[i]
        top_md = float(row["md"])
        top_tvdss = float(row["tvdss"]) if pd.notna(row["tvdss"]) else None

        if i + 1 < n:
            next_row = markers_df.iloc[i + 1]
            bottom_md = float(next_row["md"])
            bottom_tvdss = (
                float(next_row["tvdss"])
                if pd.notna(next_row["tvdss"]) else None
            )
        else:
            bottom_md = None
            bottom_tvdss = None

        # Мощность по MD
        thickness_md = (
            bottom_md - top_md if bottom_md is not None else None
        )

        # Мощность в АО: top − bottom (оба отрицательные вниз,
        # разница даёт положительное число)
        thickness_tvdss = None
        if top_tvdss is not None and bottom_tvdss is not None:
            thickness_tvdss = top_tvdss - bottom_tvdss

        layers.append({
            "name":            row["marker_name"],
            "top_md":          top_md,
            "bottom_md":       bottom_md,
            "thickness_md":    thickness_md,
            "top_tvdss":       top_tvdss,
            "bottom_tvdss":    bottom_tvdss,
            "thickness_tvdss": thickness_tvdss,
        })

    return layers
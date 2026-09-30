# -*- coding: utf-8 -*-
"""
Выгрузка результатов типизации в формат Petrel (шаг I).

Два продукта:

1. LAS с FACIES и aSP — по одной кривой на скважину.
   Диапазон — весь исходный LAS. Вне пластов — NULL.

2. Points with Attributes — по одному файлу на пласт.
   XYZ = координаты кровли из welltops. Атрибуты: Well, Layer,
   ContainerCode, Container, DominantFormCode.

Плюс общий файл легенд petrel_legends.txt.

См. docs/FEATURES.md, §9.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lasio


# Коды форм (совпадают с порядком в form_legend.png).
# non_reservoir = 0. Дальше по возрастанию «шага» в правилах.
FORM_CODES = {
    "non_reservoir":    0,
    "bell":             1,
    "funnel":           2,
    "v-shape":          3,
    "symmetric":        4,
    "cylinder":         5,
    "trapezoid-middle": 6,
    "uncertain":        7,
}

# Обратное соответствие — для легенды.
FORM_NAMES = {v: k for k, v in FORM_CODES.items()}

# Таблица транслитерации кириллицы для Petrel.
# Petrel не читает имена пластов с русскими буквами, поэтому
# при экспорте транслитерируем. Внутри проекта имена остаются
# русскими — это только для внешней системы.
_TRANSLIT = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D",
    "Е": "E", "Ё": "E", "Ж": "Zh", "З": "Z", "И": "I",
    "Й": "J", "К": "K", "Л": "L", "М": "M", "Н": "N",
    "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T",
    "У": "U", "Ф": "F", "Х": "Kh", "Ц": "Ts", "Ч": "Ch",
    "Ш": "Sh", "Щ": "Shch", "Ъ": "", "Ы": "Y", "Ь": "",
    "Э": "Eh", "Ю": "Yu", "Я": "Ya",
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
    "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "j", "к": "k", "л": "l", "м": "m", "н": "n",
    "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "",
    "э": "eh", "ю": "yu", "я": "ya",
}


def transliterate(name: str) -> str:
    """
    Транслитерировать кириллицу в латиницу для Petrel.
    Пробелы и дефисы сохраняются как есть.
    """
    return "".join(_TRANSLIT.get(ch, ch) for ch in name)

def build_facies_curve(
    dataset_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    null_value: float = -999.25,
) -> np.ndarray:
    """
    Построить кривую FACIES для одной скважины.

    Параметры
    ---------
    dataset_df : pd.DataFrame
        Из <well>/dataset.csv. Нужны колонки: depth_md, layer_name.
    intervals_df : pd.DataFrame
        Из <well>/intervals.csv. Нужны: top_md, bottom_md, kind, form_type.
    null_value : float
        Чем заполнять «нет данных» в LAS.

    Возвращает
    ----------
    np.ndarray — коды FACIES для каждой точки dataset_df.
    """
    n = len(dataset_df)
    facies = np.full(n, null_value, dtype=float)

    depth_md = dataset_df["depth_md"].to_numpy(dtype=float)

    for _, iv in intervals_df.iterrows():
        mask = (depth_md >= iv["top_md"]) & (depth_md <= iv["bottom_md"])
        if iv["kind"] == "non_reservoir":
            code = FORM_CODES["non_reservoir"]
        else:
            form = iv.get("form_type")
            code = FORM_CODES.get(form, FORM_CODES["uncertain"])
        facies[mask] = float(code)

    return facies


def export_las_facies(
    well_name: str,
    dataset_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    output_dir: Path,
    null_value: float = -999.25,
) -> Path:
    """
    Записать LAS-файл <well>_facies.las.

    Кривые: DEPT, aSP, FACIES.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth = dataset_df["depth_md"].to_numpy(dtype=float)

    # aSP: NaN → null_value
    if "aSP" in dataset_df.columns:
        asp = dataset_df["aSP"].to_numpy(dtype=float).copy()
        asp[np.isnan(asp)] = null_value
    else:
        asp = np.full_like(depth, null_value)

    facies = build_facies_curve(dataset_df, intervals_df, null_value)

    # --- Формируем LAS через lasio --------------------------------
    las = lasio.LASFile()

    las.well.WELL.value = well_name
    las.well.NULL.value = null_value
    las.well.STRT.value = float(depth.min())
    las.well.STOP.value = float(depth.max())
    if len(depth) > 1:
        las.well.STEP.value = float(np.median(np.diff(depth)))
    else:
        las.well.STEP.value = 0.1

    las.append_curve("DEPT", depth, unit="m", descr="Measured depth")
    las.append_curve("aSP", asp, unit="",
                     descr="alpha-PS (0=shale, 1=sand)")
    las.append_curve("FACIES", facies, unit="",
                     descr="Electrofacies code (see legends)")

    out_path = output_dir / f"{well_name}_facies.las"
    las.write(str(out_path), version=2.0)
    return out_path


def build_container_codes(summary_df: pd.DataFrame) -> dict[str, int]:
    """
    Присвоить сквозные коды всем уникальным container_type.

    Сортировка по алфавиту. no-sand получает свой код (не 0,
    чтобы не путать с «нет данных»).
    """
    unique = sorted(summary_df["container_type"].dropna().unique())
    return {name: i + 1 for i, name in enumerate(unique)}


def export_points_by_layer(
    summary_df: pd.DataFrame,
    markers_df: pd.DataFrame,
    container_codes: dict[str, int],
    output_dir: Path,
) -> list[Path]:
    """
    Записать по одному файлу точек на каждый пласт.

    Файл: points_<layer_name>.txt в формате Petrel Points
    with Attributes.

    Строки = скважины, где есть этот пласт.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Чистим старые файлы points_*.txt, чтобы не оставались артефакты
    # от предыдущих запусков (например, файлы со старыми именами).
    for old in output_dir.glob("points_*.txt"):
        old.unlink()

    # Индекс маркеров: (well_name, marker_name) → (X, Y, Z, TVDSS).
    # К сожалению, в markers_df нет well_name (он уже отфильтрован
    # на этапе read_markers). Значит нужен другой ключ.
    # Проще — пользоваться тем, что summary_df содержит layer_name
    # и well_name, а markers_df — marker_name, md, x, y, z.
    # Соединяем по (marker_name, md).

    # Индекс маркеров: (marker_name, round(md, 2)) → строка.
    markers_index = {}
    for _, m in markers_df.iterrows():
        key = (str(m["marker_name"]), round(float(m["md"]), 2))
        markers_index[key] = m

    written = []

    for layer in sorted(summary_df["layer_name"].unique()):
        rows = []
        sub = summary_df[summary_df["layer_name"] == layer]

        for _, row in sub.iterrows():
            # Ищем соответствующую отбивку в markers_df.
            # По имени маркера + округлённому MD.
            # В summary_df нет MD кровли пласта, но есть top_tvdss.
            # Проще искать по marker_name == layer, у которого
            # в маркерах единственная строка в этой скважине.
            # Однако в markers_df нет имени скважины…
            # Читаем top_tvdss из summary, а X, Y, Z — ищем по TVDSS.
            tvdss = float(row["top_tvdss"])
            # Ищем в markers_index любую строку с таким TVDSS.
            found = None
            for (mk, md_r), m in markers_index.items():
                if mk == layer and abs(float(m["tvdss"]) - tvdss) < 0.01:
                    found = m
                    break
            if found is None:
                # Нет отбивки — пропускаем. Это нормально:
                # в 227_1354 нет БС12, в 227_842 нет БС11-0.
                continue

            x = found.get("x", np.nan)
            y = found.get("y", np.nan)
            z = found.get("z", np.nan)
            # Если Z пуст — берём TVDSS.
            if pd.isna(z):
                z = float(found["tvdss"])

            ctype = row["container_type"]
            ccode = container_codes.get(ctype, 0)
            form = row.get("dominant_form", None)
            fcode = FORM_CODES.get(form, 7) if form is not None else 7

            rows.append({
                "x": x, "y": y, "z": z,
                "well": row["well_name"],
                "layer": layer,
                "container_code": ccode,
                "container": ctype,
                "form_code": fcode,
            })

        if not rows:
            continue

        # Запись файла.
        # Имя файла — уже транслитерированное (Petrel не читает
        # русские буквы в именах наборов).
        layer_lat = transliterate(layer)
        out_path = output_dir / f"points_{layer_lat}.txt"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Petrel Points with Attributes\n")
            f.write("# Lines starting with # are comments\n")
            f.write("VERSION 1\n")
            f.write("BEGIN HEADER\n")
            f.write("X\n")
            f.write("Y\n")
            f.write("Z\n")
            # ВАЖНО: строку "# Valid attribute types: ..." НЕ пишем.
            # Petrel её не игнорирует и создаёт лишний столбец STRING.
            # Объявления атрибутов идут сразу.
            f.write('STRING "Well"\n')
            f.write('STRING "Layer"\n')
            f.write('INT    "ContainerCode"\n')
            f.write('STRING "Container"\n')
            f.write('INT    "DominantFormCode"\n')
            f.write("END HEADER\n")
            for r in rows:
                # Layer в атрибутах тоже транслитерируем.
                f.write(
                    f"{r['x']:.2f} {r['y']:.2f} {r['z']:.2f}  "
                    f'"{r["well"]}" "{transliterate(r["layer"])}" '
                    f'{r["container_code"]} "{r["container"]}" '
                    f'{r["form_code"]}\n'
                )
        written.append(out_path)

    return written


def write_legends(
    output_path: Path,
    container_codes: dict[str, int],
) -> Path:
    """
    Записать общий файл легенд petrel_legends.txt.

    Содержит:
        • FACIES codes (0..7),
        • Container type codes (сквозные 1..N).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("FACIES / DOMINANT FORM CODES")
    lines.append("=" * 60)
    lines.append("  (совпадают с порядком в form_legend.png)")
    lines.append("")
    for code in sorted(FORM_NAMES.keys()):
        lines.append(f"  {code:>2}  = {FORM_NAMES[code]}")
    lines.append(f"  --  = нет данных (NULL = -999.25)")
    lines.append("")

    lines.append("=" * 60)
    lines.append("CONTAINER TYPE CODES")
    lines.append("=" * 60)
    lines.append("")
    for ctype, code in sorted(container_codes.items(), key=lambda x: x[1]):
        lines.append(f"  {code:>2}  = {ctype}")
    lines.append("")

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    return output_path


__all__ = [
    "FORM_CODES",
    "FORM_NAMES",
    "build_facies_curve",
    "export_las_facies",
    "build_container_codes",
    "export_points_by_layer",
    "write_legends",
]
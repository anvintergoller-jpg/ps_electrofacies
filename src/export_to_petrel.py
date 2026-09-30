# -*- coding: utf-8 -*-
"""
Экспорт результатов типизации в Petrel.

Запуск из корня проекта:
    python -m src.export_to_petrel

Читает готовые файлы из data/processed/:
    • <well>/dataset.csv     — DEPT и aSP
    • <well>/intervals.csv   — интервалы и формы
    • <well>/features.csv    — container_type по пластам
    • свежий summary_all_wells_<ts>.csv (по mtime)

Пишет в data/processed/petrel/:
    • las/<well>_facies.las
    • points/points_<layer>.txt (по одному на пласт)
    • petrel_legends.txt
"""

from pathlib import Path

import pandas as pd

from src.config_loader import load_config, load_mnemonic_map, resolve_wells
from src.ingestion.markers_reader import read_markers
from src.io.petrel_export import (
    build_container_codes,
    export_las_facies,
    export_points_by_layer,
    write_legends,
)


def print_section(text):
    line = "=" * 60
    print(f"\n{line}\n  {text}\n{line}")


def resolve_paths(well_name, cfg):
    """Пути к входным файлам одной скважины."""
    templates = cfg["paths_templates"]
    return {key: tpl.format(well=well_name) for key, tpl in templates.items()}


def find_latest_summary(processed_dir: Path) -> Path:
    """Найти самый свежий summary_all_wells_<ts>.csv."""
    candidates = sorted(
        processed_dir.glob("summary_all_wells_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Не найден summary_all_wells_*.csv в {processed_dir}"
        )
    return candidates[0]


def main():
    cfg = load_config()
    mnem = load_mnemonic_map()

    processed_dir = Path(cfg["paths"]["processed_dir"])
    out_subdir = cfg["petrel_export"]["output_subdir"]
    out_dir = processed_dir / out_subdir
    las_dir = out_dir / "las"
    points_dir = out_dir / "points"

    null_value = float(cfg["petrel_export"].get("null_value", -999.25))
    legends_name = cfg["petrel_export"].get(
        "legends_filename", "petrel_legends.txt"
    )

    print_section("ЭКСПОРТ В PETREL")

    # --- 1. LAS по каждой скважине ----------------------------------
    wells = resolve_wells(cfg)
    print(f"\n  Скважин: {len(wells)}")
    print(f"  Папка вывода: {out_dir.resolve()}")

    print("\n  LAS с FACIES:")
    for well_name in wells:
        well_dir = processed_dir / well_name
        dataset_path = well_dir / cfg["output"].get("dataset_name", "dataset.csv")
        intervals_path = well_dir / cfg["output"].get("intervals_name", "intervals.csv")

        if not dataset_path.exists() or not intervals_path.exists():
            print(f"    [!] {well_name}: нет файлов, пропуск")
            continue

        dataset_df = pd.read_csv(dataset_path, encoding="utf-8-sig")
        intervals_df = pd.read_csv(intervals_path, encoding="utf-8-sig")

        path = export_las_facies(
            well_name, dataset_df, intervals_df,
            output_dir=las_dir,
            null_value=null_value,
        )
        print(f"    {well_name}: {path.name}")

    # --- 2. Точки по пластам ----------------------------------------
    print("\n  Points with Attributes:")

    summary_path = find_latest_summary(processed_dir)
    print(f"    Сводная таблица: {summary_path.name}")
    summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")

    # Читаем отбивки (общий файл) для X, Y, Z.
    # ВАЖНО: read_markers фильтрует по well_name, а нам нужен весь
    # файл сразу. Читаем дважды — по каждой скважине — и конкатенируем.
    print("    Чтение отбивок для координат...")
    markers_all = []
    for well_name in wells:
        try:
            files = resolve_paths(well_name, cfg)
            df_m = read_markers(files["markers"], well_name, cfg["markers"])
            df_m["well_name"] = well_name
            markers_all.append(df_m)
        except Exception:
            continue
    if markers_all:
        markers_df = pd.concat(markers_all, ignore_index=True)
    else:
        markers_df = pd.DataFrame()

    container_codes = build_container_codes(summary_df)
    print(f"    Уникальных container_type: {len(container_codes)}")

    written = export_points_by_layer(
        summary_df, markers_df, container_codes, points_dir,
    )
    for p in written:
        print(f"    {p.name}")

    # --- 3. Легенды -------------------------------------------------
    legends_path = write_legends(out_dir / legends_name, container_codes)
    print(f"\n  Легенды: {legends_path.resolve()}")

    print_section("ГОТОВО")
    print(f"\n  LAS:     {las_dir.resolve()}")
    print(f"  Points:  {points_dir.resolve()}")
    print(f"  Легенды: {legends_path.name}")


if __name__ == "__main__":
    main()
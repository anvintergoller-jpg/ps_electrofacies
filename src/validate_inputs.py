# -*- coding: utf-8 -*-
"""
Батч-валидация входных данных по списку скважин.

Запуск из корня проекта:
    python -m src.validate_inputs

Список скважин берётся из config.yaml:
    • wells: "auto"   — сканируется папка LAS;
    • wells: [...]    — явный список.

Что проверяется по каждой скважине:
    1. Читается LAS (какие кривые найдены, покрытие SP и GK).
    2. Читается .dev (диапазон MD, KB).
    3. Читается общий файл отбивок и фильтруется по скважине
       (список маркеров, MD и TVDSS).
    4. Валидация: отбивки внутри LAS, попадают в интервал с SP,
       инклинометрия покрывает глубины.

Ошибка в одной скважине не останавливает остальные: скважина
помечается как "fail", обработка продолжается. В конце — сводка.
"""

from pathlib import Path

import numpy as np

from src.config_loader import load_config, load_mnemonic_map, resolve_wells
from src.ingestion.las_reader import read_las
from src.ingestion.deviation_reader import read_deviation
from src.ingestion.markers_reader import read_markers, markers_to_layers


def print_header(text):
    """Печатает заголовок секции в рамке."""
    line = "=" * 70
    print(f"\n{line}\n  {text}\n{line}")


def mask_span(values, mask):
    """По булевой маске вернуть (min, max) значений, где mask == True."""
    selected = values[mask]
    if selected.size == 0:
        return None, None
    return float(selected.min()), float(selected.max())


def resolve_paths(well_name, cfg):
    """Собрать пути к файлам одной скважины по шаблонам из config."""
    templates = cfg["paths_templates"]
    return {key: tpl.format(well=well_name) for key, tpl in templates.items()}


def validate_one_well(well_name, cfg, mnem):
    """
    Валидация одной скважины.

    Возвращает dict:
        well_name — имя
        status    — "ok" / "warnings" / "fail"
        message   — короткое описание
        layers    — список пластов (если получилось прочитать)
    """
    files = resolve_paths(well_name, cfg)

    result = {
        "well_name": well_name,
        "status": "ok",
        "message": "",
        "layers": [],
    }

    # --- 1. LAS -----------------------------------------------------
    try:
        df_las = read_las(files["las"], cfg["las"], mnem)
    except FileNotFoundError:
        result["status"] = "fail"
        result["message"] = f"LAS не найден: {files['las']}"
        return result
    except Exception as e:
        result["status"] = "fail"
        result["message"] = f"Ошибка чтения LAS: {e}"
        return result

    depth = df_las["DEPTH"].to_numpy()
    print(f"  LAS: {len(df_las)} точек, MD {depth.min():.1f}–{depth.max():.1f} м")

    for col in df_las.columns:
        if col == "DEPTH":
            continue
        values = df_las[col].to_numpy()
        mask = ~np.isnan(values)
        n_valid = int(mask.sum())
        if n_valid == 0:
            print(f"    {col}: НЕТ ДАННЫХ (все NULL)")
        else:
            lo, hi = mask_span(depth, mask)
            print(f"    {col}: покрытие {lo:.1f}–{hi:.1f} м, "
                  f"валидных {n_valid}")

    # --- 2. Отбивки (общий файл) -----------------------------------
    try:
        df_mark = read_markers(files["markers"], well_name, cfg["markers"])
    except FileNotFoundError:
        result["status"] = "fail"
        result["message"] = f"Файл отбивок не найден: {files['markers']}"
        return result
    except ValueError as e:
        result["status"] = "fail"
        result["message"] = f"Нет отбивок для скважины: {e}"
        return result
    except Exception as e:
        result["status"] = "fail"
        result["message"] = f"Ошибка чтения отбивок: {e}"
        return result

    print(f"  Отбивок: {len(df_mark)}")
    for _, row in df_mark.iterrows():
        print(f"    {row['marker_name']:>15}  "
              f"MD = {row['md']:>9.2f}  TVDSS = {row['tvdss']:>9.2f}")

    layers = markers_to_layers(df_mark)
    result["layers"] = layers
    print(f"  Пластов: {len(layers)}")

    # --- 3. Инклинометрия ------------------------------------------
    try:
        df_dev = read_deviation(files["deviation"], cfg["deviation"])
    except FileNotFoundError:
        result["status"] = "fail"
        result["message"] = f".dev не найден: {files['deviation']}"
        return result
    except Exception as e:
        result["status"] = "fail"
        result["message"] = f"Ошибка чтения .dev: {e}"
        return result

    md_dev = df_dev["MD"].to_numpy()
    print(f"  .dev: {len(df_dev)} точек, MD {md_dev.min():.1f}–{md_dev.max():.1f} м")
    kb = df_dev.attrs.get("kb")
    if kb is not None:
        print(f"    KB = {kb} м")

    # --- 4. Проверки ------------------------------------------------
    warnings = []

    md_mark = df_mark["md"].to_numpy()
    las_min, las_max = float(depth.min()), float(depth.max())

    # 4.1. Отбивки внутри LAS
    if md_mark.min() < las_min or md_mark.max() > las_max:
        warnings.append(
            f"отбивки выходят за LAS ({las_min:.1f}–{las_max:.1f})"
        )

    # 4.2. Отбивки в интервале, где SP не NULL
    if "SP" in df_las.columns:
        sp = df_las["SP"].to_numpy()
        mask_sp = ~np.isnan(sp)
        if mask_sp.any():
            sp_lo, sp_hi = mask_span(depth, mask_sp)
            if md_mark.min() < sp_lo or md_mark.max() > sp_hi:
                warnings.append(
                    f"не все отбивки в интервале с SP "
                    f"({sp_lo:.1f}–{sp_hi:.1f})"
                )
        else:
            warnings.append("кривая SP полностью NULL")
            result["status"] = "fail"

    # 4.3. Инклинометрия покрывает отбивки
    if md_mark.max() > md_dev.max():
        warnings.append(
            f"отбивки уходят глубже .dev "
            f"({md_mark.max():.1f} > {md_dev.max():.1f})"
        )

    if warnings:
        if result["status"] != "fail":
            result["status"] = "warnings"
        result["message"] = "; ".join(warnings)

    return result


def main():
    cfg = load_config()
    mnem = load_mnemonic_map()

    try:
        wells = resolve_wells(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  [!] Не удалось получить список скважин: {e}")
        return

    if not wells:
        print("  Список скважин пуст.")
        return

    print_header(f"БАТЧ-ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ — {len(wells)} скважин")
    print(f"  Режим wells: {cfg.get('wells')!r}")
    print(f"  Скважины: {', '.join(wells)}")

    results = []
    for i, well_name in enumerate(wells, start=1):
        print_header(f"[{i}/{len(wells)}] СКВАЖИНА {well_name}")
        try:
            r = validate_one_well(well_name, cfg, mnem)
        except Exception as e:
            r = {
                "well_name": well_name,
                "status": "fail",
                "message": f"непредвиденная ошибка: {e}",
                "layers": [],
            }
        results.append(r)
        print(f"  → Статус: {r['status'].upper()}"
              + (f" ({r['message']})" if r["message"] else ""))

    # --- Сводка -----------------------------------------------------
    print_header("СВОДКА")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_warn = sum(1 for r in results if r["status"] == "warnings")
    n_fail = sum(1 for r in results if r["status"] == "fail")

    for r in results:
        icon = {"ok": "✓", "warnings": "⚠", "fail": "✗"}[r["status"]]
        msg = f" — {r['message']}" if r["message"] else ""
        print(f"  {icon} {r['well_name']:<16} {r['status']:<10}{msg}")

    print(f"\n  Итого: {n_ok} ок, {n_warn} с замечаниями, {n_fail} сбоев.")

    if n_fail > 0:
        print("\n  Сбойные скважины — проверьте пути к файлам и наличие")
        print("  отбивок для них в общем файле welltops_all.xlsx.")


if __name__ == "__main__":
    main()
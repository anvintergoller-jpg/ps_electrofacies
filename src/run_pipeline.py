# -*- coding: utf-8 -*-
"""
Батч-обработка скважин.

Запуск из корня проекта:
    python -m src.run_pipeline

Логика:
    1. Читаем config, получаем список скважин (resolve_wells).
    2. По каждой скважине — process_well(): полный цикл
       (LAS → датасет → сегментация → форма → размер/положение →
        признаки → container_type → PNG).
    3. Результат — в data/processed/<well_name>/:
           dataset.csv, intervals.csv, features.csv, well_log.png, log.txt
    4. Сводка по всем скважинам — data/processed/summary_all_wells_<ts>.csv
    5. Общий лог запуска — data/processed/run_log_<ts>.txt
    6. Легенда форм (общая) — data/processed/form_legend.png
    7. Ошибка в одной скважине не останавливает остальные.

Список скважин берётся из config.yaml, секция wells:
    wells: "auto"       — сканируем data/raw/las/*.las
    wells: [список]     — только указанные скважины
"""

from pathlib import Path
from datetime import datetime

import io
import contextlib
import traceback

import numpy as np
import pandas as pd

from src.config_loader import load_config, load_mnemonic_map, resolve_wells
from src.ingestion.las_reader import read_las
from src.ingestion.deviation_reader import read_deviation, md_to_tvd
from src.ingestion.markers_reader import read_markers
from src.preprocessing.normalization import normalize_curves
from src.application.build_dataset import build_dataset
from src.domain.features import compute_container_features
from src.domain.segmentation import segment_all_layers
from src.domain.classify_form import ClassificationParams, classify_intervals
from src.domain.size_class import classify_size_and_position
from src.domain.container_type import compute_container_types
from src.visualization.well_log import plot_well
from src.visualization.form_legend import plot_form_legend
from src.domain.cross_well_summary import build_cross_well_summary
from src.domain.elements_v3 import split_interval_into_elements
from src.domain.classify_form_v3 import classify_form_v3

def print_section(text):
    """Печатает заголовок секции в рамке."""
    line = "=" * 60
    print(f"\n{line}\n  {text}\n{line}")


def resolve_paths(well_name, cfg):
    """
    Собрать пути к файлам одной скважины по шаблонам из config.
    Возвращает dict с ключами las, deviation, markers.
    """
    templates = cfg["paths_templates"]
    return {key: tpl.format(well=well_name) for key, tpl in templates.items()}


def process_well(well_name, cfg, mnem, output_dir):
    """
    Полный цикл обработки одной скважины.

    Пишет результат в output_dir/<well_name>/:
        dataset.csv, intervals.csv, features.csv, well_log.png

    Возвращает dict:
        well_name   — имя скважины
        status      — "ok" (при сбое main() обернёт в fail)
        features_df — DataFrame признаков (для сводки) или None
        files       — dict с путями сохранённых файлов
    """
    files = resolve_paths(well_name, cfg)

    print_section(f"ОБРАБОТКА СКВАЖИНЫ {well_name}")

    # --- 1. Чтение входных данных -----------------------------------
    print_section("Чтение LAS")
    df_las = read_las(files["las"], cfg["las"], mnem)
    print(f"  Точек: {len(df_las)}")

    print_section("Чтение инклинометрии")
    df_dev = read_deviation(files["deviation"], cfg["deviation"])
    print(f"  Точек: {len(df_dev)}")

    print_section("Чтение отбивок")
    df_mark = read_markers(files["markers"], well_name, cfg["markers"])
    print(f"  Маркеров: {len(df_mark)}")

    # --- 2. Рабочий интервал ----------------------------------------
    top_md = float(df_mark["md"].min())
    bottom_md = float(df_dev["MD"].max())

    print_section("Рабочий интервал")
    print(f"  Верх (первая кровля): {top_md:.2f} м MD")
    print(f"  Низ  (конец .dev):    {bottom_md:.2f} м MD")
    print(f"  Мощность интервала:   {bottom_md - top_md:.2f} м")

    # --- 3. Нормализация --------------------------------------------
    print_section("Нормализация кривых")
    df_las = normalize_curves(
        df_las, cfg["normalization"], interval=(top_md, bottom_md),
    )
    bl = df_las.attrs.get("baselines", {})
    for name, (shale, sand) in bl.items():
        print(f"  {name}: shale = {shale:.2f}, sand = {sand:.2f}")

    # --- 4. Сборка датасета -----------------------------------------
    print_section("Сборка единой таблицы")
    df = build_dataset(df_las, df_dev, df_mark, cfg, well_name)
    print(f"  Строк всего: {len(df)}")
    print(f"  Колонки: {list(df.columns)}")

    # Замыкание: MD → TVDSS (Petrel-стиль, отрицательное вниз).
    _kb = df_dev.attrs.get("kb", 0.0)

    def md_to_tvdss(md_value):
        tvd = float(md_to_tvd(df_dev, np.array([md_value]))[0])
        return _kb - tvd

    # Папка вывода по скважине.
    well_dir = output_dir / well_name
    well_dir.mkdir(parents=True, exist_ok=True)

    # --- 5. Сохранение датасета -------------------------------------
    dataset_name = cfg["output"].get("dataset_name", "dataset.csv")
    dataset_path = well_dir / dataset_name
    df.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    print(f"  Датасет: {dataset_path.resolve()}")

    # --- 6. Сегментация ---------------------------------------------
    print_section("Сегментация пластов")
    intervals_df = segment_all_layers(
        df,
        df.attrs["layers"],
        md_to_tvdss_fn=md_to_tvdss,
        cutoff=cfg["reservoir"]["cutoff"],
        smooth_window=cfg["preprocessing"]["smooth_window"],
        min_thickness_tvdss=cfg["reservoir"]["min_thickness_tvdss"],
    )
    n_res = int((intervals_df["kind"] == "reservoir").sum()) \
        if not intervals_df.empty else 0
    n_non = int((intervals_df["kind"] == "non_reservoir").sum()) \
        if not intervals_df.empty else 0
    print(f"  Всего интервалов: {len(intervals_df)} "
          f"(reservoir: {n_res}, non_reservoir: {n_non})")

    # --- 7. Классификация формы -------------------------------------
    print_section("Классификация формы аномалий ПС")

    # Параметры классификатора собираются из двух секций config:
    #   • classification — правила формы
    #   • reservoir     — общий порог коллектора (одинаковый
    #                     с сегментацией)
    cls_cfg = dict(cfg["classification"])
    cls_cfg["reservoir_cutoff"] = cfg["reservoir"]["cutoff"]
    cls_params = ClassificationParams.from_dict(cls_cfg)
    print(f"  Правила: версия {cls_params.rules_version}")

    intervals_df = classify_intervals(
        df, intervals_df, params=cls_params,
        smooth_window=cfg["preprocessing"]["smooth_window"],
    )
    res_only = intervals_df[intervals_df["kind"] == "reservoir"]
    if not res_only.empty:
        print("  Формы:")
        for ft, cnt in res_only["form_type"].value_counts().items():
            print(f"    {ft:<17} {cnt}")

    
    # --- 7b. ПРОБНЫЙ ПРОГОН v3.0 (разбиение на элементы) ------------
    # Только для пилота: печатаем в лог и сохраняем отдельным файлом
    # intervals_v3.csv. Стандартный intervals.csv не трогаем.
    # При ошибке — печатаем traceback, но не валим пайплайн.
    print_section("ПРОБНЫЙ ПРОГОН v3.0 (разбиение на элементы)")

    if "aSP_disc" not in df.columns:
        print("  [!] Колонка aSP_disc не найдена — пропускаем.")
        print("      Примените патч aSP_disc и перезапустите.")
    else:
        try:
            from src.domain.elements_v3 import split_interval_into_elements

            v3_params = cfg["classification_v3"]
            smooth_window = cfg["preprocessing"]["smooth_window"]

            depth_md_arr = df["depth_md"].to_numpy(dtype=float)
            res_intervals = intervals_df[
                intervals_df["kind"] == "reservoir"
            ]
            print(f"  Reservoir-интервалов: {len(res_intervals)}")

            v3_rows = []
            for _, iv in res_intervals.iterrows():
                mask = (
                    (depth_md_arr >= iv["top_md"])
                    & (depth_md_arr <= iv["bottom_md"])
                )
                df_iv = df.loc[mask]
                if df_iv.empty:
                    continue

                asp_slice = df_iv["aSP"].to_numpy(dtype=float)
                depth_slice = df_iv["depth_tvdss"].to_numpy(dtype=float)
                asp_disc_slice = df_iv["aSP_disc"].to_numpy(dtype=float)

                elem = split_interval_into_elements(
                    asp_slice, depth_slice, asp_disc_slice,
                    params=v3_params,
                    smooth_window=smooth_window,
                )

                # --- Классификация формы v3.0 ---------------------
                form_info = classify_form_v3(elem, v3_params)
                elem.update(form_info)

                # Компактная печать одной строкой.
                if elem["has_plateau"]:
                    plat = (f"{elem['plateau_dominant_class']} "
                            f"{elem['plateau_length_m']:.2f}м "
                            f"({elem['plateau_length_frac']*100:.0f}%) "
                            f"asp={elem['plateau_mean_asp']:.2f} "
                            f"{elem['plateau_trend']}")
                else:
                    mp = elem["max_position"]
                    plat = (f"— max@{mp:.2f}" if mp is not None else "—")

                top_s = (f"{elem['top_type']} "
                         f"{elem['top_length_m']:.2f}м "
                         f"({elem['top_length_frac']*100:.0f}%)")
                bot_s = (f"{elem['bot_type']} "
                         f"{elem['bot_length_m']:.2f}м "
                         f"({elem['bot_length_frac']*100:.0f}%)")

                print(f"    {iv['layer_name']:<15} "
                      f"[{iv['interval_index']}] "
                      f"{iv['thickness_tvdss']:.2f}м  "
                      f"Кр={top_s}  Пл={plat}  Пд={bot_s}  "
                      f"n_pl={elem['n_plateaus']}  "
                      f"→ {elem['form_type']} "
                      f"({elem['confidence']:.2f})")

                # Полная запись — в CSV.
                v3_rows.append({
                    "layer_name": iv["layer_name"],
                    "interval_index": iv["interval_index"],
                    **elem,
                })

            # Сохраняем полный результат отдельным файлом.
            if v3_rows:
                v3_df = pd.DataFrame(v3_rows)
                v3_path = well_dir / "intervals_v3.csv"
                v3_df.to_csv(v3_path, index=False, encoding="utf-8-sig")
                print(f"  v3.0 сохранён: {v3_path.resolve()}")

                # --- Подмена form_type на v3.0 -------------------
                # Заменяем форму и confidence в intervals_df на v3.0
                # для тех интервалов, что попали в пробный прогон.
                # Не-резервуары и интервалы без v3 остаются как были.
                # Это позволяет container_type.py и petrel_export.py
                # работать с v3-формами, не меняя свой код.
                                # Колонки, которые нужно пробросить из v3 в основной
                # intervals_df. Формы и confidence — ключевые.
                # Остальные — для визуализации и отчёта.
                v3_probe_cols = [
                    "form_type", "confidence",
                    "symmetry_top_bot",
                    "plateau_length_m", "plateau_mean_asp",
                    "plateau_trend",
                ]
                v3_probe = v3_df[
                    ["layer_name", "interval_index"] + v3_probe_cols
                ].copy()
                # Суффикс _v3, чтобы не столкнуться с существующими
                # колонками intervals_df (в т.ч. form_type от v2).
                rename_map = {
                    c: f"_v3_{c}" for c in v3_probe_cols
                }
                v3_probe = v3_probe.rename(columns=rename_map)

                intervals_df = intervals_df.merge(
                    v3_probe,
                    on=["layer_name", "interval_index"],
                    how="left",
                )

                mask_v3 = intervals_df["_v3_form_type"].notna()

                # Заменяем старые значения на v3 там, где они есть.
                # form_type и confidence — прямо.
                intervals_df.loc[mask_v3, "form_type"] = \
                    intervals_df.loc[mask_v3, "_v3_form_type"]
                intervals_df.loc[mask_v3, "confidence"] = \
                    intervals_df.loc[mask_v3, "_v3_confidence"]

                # Остальные колонки — создаём новые в intervals_df,
                # если их там ещё нет.
                for col in v3_probe_cols:
                    if col in ("form_type", "confidence"):
                        continue
                    new_col = col
                    if new_col not in intervals_df.columns:
                        intervals_df[new_col] = None
                    intervals_df.loc[mask_v3, new_col] = \
                        intervals_df.loc[mask_v3, f"_v3_{col}"]

                # Убираем служебные _v3_* колонки.
                drop_cols = [
                    f"_v3_{c}" for c in v3_probe_cols
                    if f"_v3_{c}" in intervals_df.columns
                ]
                intervals_df = intervals_df.drop(columns=drop_cols)

                print(f"  form_type заменён на v3.0: "
                      f"{int(mask_v3.sum())} интервалов")

        except Exception as e:
            # Не валим пайплайн из-за пилота.
            import traceback as _tb
            print(f"  [!] Ошибка пробного прогона v3.0: {e}")
            print(_tb.format_exc())

    # --- 8. Размер и положение --------------------------------------
    print_section("Категории размера и положение в контейнере")
    intervals_df = classify_size_and_position(
        df, intervals_df, df.attrs["layers"], cfg["size_classification"],
    )
    print(f"  Интервалов с формой и размером: {len(intervals_df)}")

    # Сохранение интервалов.
    intervals_name = cfg["output"].get("intervals_name", "intervals.csv")
    intervals_path = well_dir / intervals_name
    intervals_df.to_csv(intervals_path, index=False, encoding="utf-8-sig")
    print(f"  Интервалы: {intervals_path.resolve()}")

        # --- 9. Признаки уровня A ---------------------------------------
    print_section("Признаки пластов (уровень A)")
    features_df = compute_container_features(
        df, df.attrs["layers"],
        smooth_window=cfg["preprocessing"]["smooth_window"],
    )
    print(f"  Пластов: {len(features_df)}")

    # --- 10. Сводная типизация пласта (уровень C) -------------------
    print_section("Сводная типизация пластов")
    features_df = compute_container_types(
        features_df, intervals_df, df.attrs["layers"], cfg["container_type"],
    )
    for _, r in features_df.iterrows():
        print(f"    {r['layer_name']:<16} {r['container_type']}")

    # Сохранение признаков.
    features_name = cfg["output"].get("features_name", "features.csv")
    features_path = well_dir / features_name
    features_df.to_csv(features_path, index=False, encoding="utf-8-sig")
    print(f"  Признаки: {features_path.resolve()}")

    # --- 11. Визуализация -------------------------------------------
    print_section("Визуализация")
    kb = df_dev.attrs.get("kb", 0.0)
    tvd_top = float(md_to_tvd(df_dev, np.array([top_md]))[0])
    tvd_bot = float(md_to_tvd(df_dev, np.array([bottom_md]))[0])
    tvdss_top = kb - tvd_top
    tvdss_bot = kb - tvd_bot
    print(f"  Рабочий интервал в TVDSS: {tvdss_top:.1f} … {tvdss_bot:.1f} м")

    title = (f"Скважина {well_name}: SP + aSP, GK, отбивки\n"
             f"АО {tvdss_top:.0f} … {tvdss_bot:.0f} м")

    plot_name = cfg["output"].get("plot_name", "well_log.png")
    png_path = well_dir / plot_name
    plot_well(
        df,
        title=title,
        output_path=png_path,
        reservoir_cutoff=cfg["reservoir"]["cutoff"],
        tvdss_range=(tvdss_top, tvdss_bot),
        layers=df.attrs.get("layers", []),
        intervals_df=intervals_df,
        classification_params=cls_params,
    )

    return {
        "well_name": well_name,
        "status": "ok",
        "features_df": features_df,
        "files": {
            "dataset": dataset_path,
            "intervals": intervals_path,
            "features": features_path,
            "plot": png_path,
        },
    }


def main():
    # --- 1. Конфиги -------------------------------------------------
    cfg = load_config()
    mnem = load_mnemonic_map()

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- 2. Список скважин ------------------------------------------
    try:
        wells = resolve_wells(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"Ошибка получения списка скважин: {e}")
        return

    if not wells:
        print("Список скважин пуст.")
        return

    # --- 3. Подготовка папки вывода ---------------------------------
    output_dir = Path(cfg["paths"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Легенду форм рисуем один раз — она не зависит от скважины.
    legend_path = output_dir / "form_legend.png"
    plot_form_legend(legend_path, cutoff=cfg["reservoir"]["cutoff"])

    print_section(f"БАТЧ-ОБРАБОТКА — {len(wells)} скважин")
    print(f"  Режим wells: {cfg.get('wells')!r}")
    print(f"  Метка запуска: {run_id}")
    print(f"  Папка вывода: {output_dir.resolve()}")

    results = []
    all_features = []
    run_log_lines = []

    # --- 4. Цикл по скважинам ---------------------------------------
    for i, well_name in enumerate(wells, start=1):
        print_section(f"[{i}/{len(wells)}] {well_name}")

        # Всё, что печатает process_well, перехватываем в буфер.
        # В консоль пойдёт только короткая строка отчёта.
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                result = process_well(well_name, cfg, mnem, output_dir)
        except Exception as e:
            result = {
                "well_name": well_name,
                "status": "fail",
                "message": str(e),
                "features_df": None,
                "files": {},
                "traceback": traceback.format_exc(),
            }

        well_log_text = buf.getvalue()

        # Сохраняем лог по скважине (даже если была ошибка).
        well_dir = output_dir / well_name
        well_dir.mkdir(parents=True, exist_ok=True)
        log_name = cfg["output"].get("log_name", "log.txt")
        log_path = well_dir / log_name
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(well_log_text)
            if result["status"] == "fail":
                f.write("\n\n--- TRACEBACK ---\n")
                f.write(result.get("traceback", ""))

        # Общий лог: накапливаем.
        run_log_lines.append(
            f"\n{'=' * 70}\n  СКВАЖИНА {well_name}\n{'=' * 70}\n"
        )
        run_log_lines.append(well_log_text)
        if result["status"] == "fail":
            run_log_lines.append("\n--- ОШИБКА ---\n")
            run_log_lines.append(result.get("traceback", ""))

        # Короткий отчёт в консоль.
        if result["status"] == "ok":
            fdf = result["features_df"]
            n_layers = len(fdf) if fdf is not None else 0
            if fdf is not None and "ntg_tvdss" in fdf.columns:
                n_with_sand = int((fdf["ntg_tvdss"] > 0).sum())
            else:
                n_with_sand = 0
            print(f"  ✓ {well_name}: {n_layers} пластов, "
                  f"{n_with_sand} с коллекторами")
            print(f"    лог: {log_path.resolve()}")
        else:
            print(f"  ✗ {well_name}: сбой — {result['message']}")
            print(f"    лог: {log_path.resolve()}")

        results.append(result)

        if result["features_df"] is not None:
            df_w = result["features_df"].copy()
            df_w.insert(0, "well_name", well_name)
            all_features.append(df_w)

    # --- 5. Сводная таблица по всем скважинам ------------------------
    summary_template = cfg["output"].get(
        "summary_name", "summary_all_wells_{ts}.csv"
    )
    summary_name = summary_template.format(ts=run_id)
    summary_path = output_dir / summary_name

    if all_features:
        summary_df = pd.concat(all_features, ignore_index=True)
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print_section("СВОДНАЯ ТАБЛИЦА")
        print(f"  Строк (пластов × скважин): {len(summary_df)}")
        print(f"  Файл: {summary_path.resolve()}")

        # --- Сопоставление пластов между скважинами (шаг H) ---------
        print_section("СОПОСТАВЛЕНИЕ ПЛАСТОВ МЕЖДУ СКВАЖИНАМИ")
        cross_well_paths = build_cross_well_summary(
            summary_df,
            output_dir,
            cfg.get("cross_well_summary", {}),
        )
        print(f"\n  Артефакты:")
        for name, path in cross_well_paths.items():
            print(f"    {name:<16} {path.resolve()}")
    else:
        print("\n  [!] Нет данных для сводной таблицы — все скважины упали.")

    # --- 6. Общий лог запуска ---------------------------------------
    run_log_template = cfg["output"].get(
        "run_log_name", "run_log_{ts}.txt"
    )
    run_log_name = run_log_template.format(ts=run_id)
    run_log_path = output_dir / run_log_name
    with open(run_log_path, "w", encoding="utf-8") as f:
        f.writelines(run_log_lines)

    # --- 7. Итоговая сводка -----------------------------------------
    print_section("ИТОГО")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    print(f"  Обработано: {len(results)}")
    print(f"  Успешно:    {n_ok}")
    print(f"  Сбоев:      {n_fail}")
    print(f"\n  Сводная таблица: {summary_path.resolve()}")
    print(f"  Общий лог:       {run_log_path.resolve()}")
    print(f"  Легенда форм:    {legend_path.resolve()}")


if __name__ == "__main__":
    main()
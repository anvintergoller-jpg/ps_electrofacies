"""
Полный цикл обработки одной скважины.

Запуск из корня проекта:
    python -m src.run_pipeline

Шаги:
    1. Читаем конфиги.
    2. Читаем LAS, .dev, отбивки.
    3. Определяем рабочий интервал.
    4. Нормализуем кривые внутри рабочего интервала.
    5. Собираем единый DataFrame.
    6. Сохраняем его в data/interim/.
    7. Считаем признаки пластов (уровень A).
    8. Строим график и сохраняем в results/plots/.

Выходные файлы:
    data/interim/{well_name}_dataset.csv
    data/interim/{well_name}_features.csv
    results/plots/{well_name}_well_log.png
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.config_loader import load_config, load_mnemonic_map
from src.ingestion.las_reader import read_las
from src.ingestion.deviation_reader import read_deviation, md_to_tvd
from src.ingestion.markers_reader import read_markers
from src.preprocessing.normalization import normalize_curves
from src.application.build_dataset import build_dataset
from src.domain.features import compute_container_features
from src.visualization.well_log import plot_well
from src.domain.segmentation import segment_all_layers



def print_section(text):
    """Печатает заголовок секции в рамке."""
    line = "=" * 60
    print(f"\n{line}\n  {text}\n{line}")


def main():
    # --- 1. Конфиги -------------------------------------------------
    cfg = load_config()
    mnem = load_mnemonic_map()

    well_cfg = cfg["pilot_well"]
    well_name = well_cfg["name"]
    files = well_cfg["files"]

    print_section(f"ОБРАБОТКА СКВАЖИНЫ {well_name}")

    # Метка запуска — используется в именах выходных файлов
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"  Метка запуска: {run_id}")

    # --- 2. Чтение входных данных -----------------------------------
    print_section("Чтение LAS")
    df_las = read_las(files["las"], cfg["las"], mnem)
    print(f"  Точек: {len(df_las)}")

    print_section("Чтение инклинометрии")
    df_dev = read_deviation(files["deviation"], cfg["deviation"])
    print(f"  Точек: {len(df_dev)}")

    print_section("Чтение отбивок")
    df_mark = read_markers(files["markers"], well_name, cfg["markers"])
    print(f"  Маркеров: {len(df_mark)}")

    # --- 3. Определяем рабочий интервал -----------------------------
    # Верх — самая верхняя кровля, низ — конец данных в .dev
    top_md = float(df_mark["md"].min())
    bottom_md = float(df_dev["MD"].max())

    print_section("Рабочий интервал")
    print(f"  Верх (первая кровля): {top_md:.2f} м MD")
    print(f"  Низ  (конец .dev):    {bottom_md:.2f} м MD")
    print(f"  Мощность интервала:   {bottom_md - top_md:.2f} м")

    # --- 4. Нормализация внутри интервала ---------------------------
    print_section("Нормализация кривых (внутри интервала)")
    df_las = normalize_curves(
        df_las,
        cfg["normalization"],
        interval=(top_md, bottom_md),
    )
    bl = df_las.attrs.get("baselines", {})
    for name, (shale, sand) in bl.items():
        print(f"  {name}: shale = {shale:.2f}, sand = {sand:.2f}")

    # --- 5. Сборка единого набора -----------------------------------
    print_section("Сборка единой таблицы")
    df = build_dataset(df_las, df_dev, df_mark, cfg)
    print(f"  Строк всего: {len(df)}")
    print(f"  Колонки: {list(df.columns)}")

    n_in = int(((df["depth_md"] >= top_md) & (df["depth_md"] <= bottom_md)).sum())
    print(f"  Строк в рабочем интервале: {n_in}")

    print("\n  Точек в каждом пласте:")
    layer_counts = df["layer_name"].value_counts(dropna=False)
    for name, count in layer_counts.items():
        label = name if name is not None else "(вне пластов)"
        print(f"    {label:>15}: {count}")
    
    # Замыкание: MD → TVDSS. Оборачиваем для передачи в сегментацию.
    _kb = df_dev.attrs.get("kb", 0.0)
    def md_to_tvdss(md_value):
        tvd = float(md_to_tvd(df_dev, np.array([md_value]))[0])
        return _kb - tvd

     # --- 6. Сохранение в interim ------------------------------------
    print_section("Сохранение результата")
    interim_dir = Path(cfg["paths"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    csv_path = interim_dir / f"{well_name}_dataset_{run_id}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Таблица: {csv_path.resolve()}")

    # --- 7. Сегментация пластов -------------------------------------
    print_section("Сегментация пластов")

    intervals_df = segment_all_layers(
        df,
        df.attrs["layers"],
        md_to_tvdss_fn=md_to_tvdss,
        cfg=cfg["segmentation"],
    )

    # Сводка по типам интервалов
    print(f"\n  Всего интервалов: {len(intervals_df)}")
    if not intervals_df.empty:
        kind_counts = intervals_df["kind"].value_counts()
        for kind, count in kind_counts.items():
            print(f"    {kind}: {count}")

        print("\n  Интервалы по пластам (мощности в АО):")
        header = (f"  {'Пласт':<16} {'№':>3} {'Тип':<15} "
                  f"{'Кровля АО':>10} {'Подошва АО':>11} "
                  f"{'Мощн.AO':>8} {'Точек':>7}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, r in intervals_df.iterrows():
            print(f"  {r['layer_name']:<16} "
                  f"{r['interval_index']:>3d} "
                  f"{r['kind']:<15} "
                  f"{r['top_tvdss']:>10.2f} "
                  f"{r['bottom_tvdss']:>11.2f} "
                  f"{r['thickness_tvdss']:>8.2f} "
                  f"{r['n_points']:>7d}")

    # Сохраняем интервалы в interim
    intervals_path = interim_dir / f"{well_name}_intervals_{run_id}.csv"
    intervals_df.to_csv(intervals_path, index=False, encoding="utf-8-sig")
    print(f"\n  Интервалы: {intervals_path.resolve()}")

    # --- 8. Признаки пластов (уровень A) ----------------------------
    print_section("Признаки пластов (уровень A)")
    features_df = compute_container_features(df, df.attrs["layers"])

    print("\n  Признаки пласта-контейнера (мощности в АО):")
    header = (f"  {'Пласт':<16} {'Мощн.AO':>9} {'Мощн.MD':>9} "
              f"{'Точек':>7} {'SP valid':>9} {'Покрытие':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in features_df.iterrows():
        th_ao = r["thickness_tvdss"] if pd.notna(r["thickness_tvdss"]) else 0
        th_md = r["thickness_md"] if pd.notna(r["thickness_md"]) else 0
        print(f"  {r['layer_name']:<16} "
              f"{th_ao:>9.2f} "
              f"{th_md:>9.2f} "
              f"{r['n_points']:>7d} "
              f"{r['n_sp_valid']:>9d} "
              f"{r['sp_coverage']:>9.2f}")
        
        # Литологический профиль — отдельной таблицей
    print("\n  Литологический профиль пластов (доля точек в каждом классе αПС):")
    header2 = (f"  {'Пласт':<16} "
               f"{'глина':>7} {'ал.глин':>8} {'алевр':>7} "
               f"{'пес.ср':>7} {'пес.кр':>7}")
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for _, r in features_df.iterrows():
        print(f"  {r['layer_name']:<16} "
              f"{r['pct_shale']:>7.2f} "
              f"{r['pct_silty_shale']:>8.2f} "
              f"{r['pct_silt']:>7.2f} "
              f"{r['pct_sand_m']:>7.2f} "
              f"{r['pct_sand_c']:>7.2f}")

    features_path = interim_dir / f"{well_name}_features_{run_id}.csv"
    features_df.to_csv(features_path, index=False, encoding="utf-8-sig")
    print(f"\n  Признаки: {features_path.resolve()}")

    # --- 9. Визуализация рабочего интервала -------------------------
    print_section("Визуализация")
    plots_dir = Path(cfg["paths"]["results_dir"]) / "plots"
    png_path = plots_dir / f"{well_name}_well_log_{run_id}.png"

    # Интервал в TVDSS. Используем простую интерполяцию по .dev,
    # затем переводим в Petrel-стиль: TVDSS = KB − TVD.
    kb = df_dev.attrs.get("kb", 0.0)

    tvd_top = float(md_to_tvd(df_dev, np.array([top_md]))[0])
    tvd_bot = float(md_to_tvd(df_dev, np.array([bottom_md]))[0])
    tvdss_top = kb - tvd_top
    tvdss_bot = kb - tvd_bot

    print(f"  Рабочий интервал в TVDSS: "
          f"{tvdss_top:.1f} … {tvdss_bot:.1f} м")

    title = (f"Скважина {well_name}: SP, GK, отбивки\n"
             f"АО {tvdss_top:.0f} … {tvdss_bot:.0f} м")

    plot_well(
        df,
        title=title,
        output_path=png_path,
        tvdss_range=(tvdss_top, tvdss_bot),
        layers=df.attrs.get("layers", []),
        intervals_df=intervals_df,
    )

    print_section("ГОТОВО")
    print(f"\n  Файлы запуска {run_id}:")
    print(f"    {csv_path.name}")
    print(f"    {features_path.name}")
    print(f"    {png_path.name}")


if __name__ == "__main__":
    main()
"""
Валидация входных данных для пилотной скважины.

Запуск из корня проекта:
    python -m src.validate_inputs

Что делает:
    1. Читает config.yaml и mnemonic_map.yaml.
    2. Читает LAS, .dev, отбивки для пилотной скважины.
    3. Печатает сводку: диапазоны глубин, покрытие кривых, список маркеров.
    4. Делает проверки: попадают ли отбивки в интервал LAS, где SP не NULL,
       покрывает ли инклинометрия все нужные глубины, и т.д.
"""

from pathlib import Path

from src.config_loader import load_config, load_mnemonic_map
from src.ingestion.las_reader import read_las
from src.ingestion.deviation_reader import read_deviation
from src.ingestion.markers_reader import read_markers, markers_to_layers


# ----------------------------------------------------------------------
# Небольшие функции-помощники для печати
# ----------------------------------------------------------------------

def print_header(text):
    """Печатает заголовок секции в рамке."""
    line = "=" * 60
    print(f"\n{line}\n  {text}\n{line}")


def mask_span(values, mask):
    """
    По булевой маске возвращает (min, max) тех значений, где mask == True.
    Если таких нет — возвращает (None, None).
    """
    selected = values[mask]
    if selected.size == 0:
        return None, None
    return float(selected.min()), float(selected.max())


# ----------------------------------------------------------------------
# Основная программа
# ----------------------------------------------------------------------

def main():
    # --- 1. Конфиги ---------------------------------------------------
    cfg = load_config()
    mnem = load_mnemonic_map()

    well = cfg["pilot_well"]
    well_name = well["name"]
    files = well["files"]

    print_header(f"ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ — скважина {well_name}")

    # --- 2. LAS -------------------------------------------------------
    print_header("LAS (каротаж)")
    print(f"  Файл: {files['las']}")
    df_las = read_las(files["las"], cfg["las"], mnem)

    depth = df_las["DEPTH"].to_numpy()
    print(f"  Точек: {len(df_las)}")
    print(f"  Глубины (MD): {depth.min():.1f} – {depth.max():.1f} м")

    # Покрытие каждой кривой: где значение не NaN
    for col in df_las.columns:
        if col == "DEPTH":
            continue
        values = df_las[col].to_numpy()
        mask = ~np.isnan(values)
        n_valid = mask.sum()
        if n_valid == 0:
            print(f"    {col}: НЕТ ДАННЫХ (все NULL)")
            continue
        lo, hi = mask_span(depth, mask)
        print(f"    {col}: покрытие {lo:.1f} – {hi:.1f} м, "
              f"валидных точек {n_valid}")

    # --- 3. Инклинометрия --------------------------------------------
    print_header("Инклинометрия (.dev)")
    print(f"  Файл: {files['deviation']}")
    df_dev = read_deviation(files["deviation"], cfg["deviation"])

    md_dev = df_dev["MD"].to_numpy()
    tvd_dev = df_dev["TVD"].to_numpy()
    print(f"  Точек: {len(df_dev)}")
    print(f"  MD:  {md_dev.min():.1f} – {md_dev.max():.1f} м")
    print(f"  TVD: {tvd_dev.min():.1f} – {tvd_dev.max():.1f} м")
    print(f"  Смещение MD − TVD на забое: {md_dev.max() - tvd_dev.max():.1f} м")

    # --- 4. Отбивки ---------------------------------------------------
    print_header("Отбивки (WellTops)")
    print(f"  Файл: {files['markers']}")
    df_mark = read_markers(files["markers"], well_name, cfg["markers"])
    print(f"  Маркеров: {len(df_mark)}")
    for _, row in df_mark.iterrows():
        print(f"    {row['marker_name']:>15}  MD = {row['md']:>9.2f}  "
              f"TVDSS = {row['tvdss']:>9.2f}")

    # Список пластов: между соседними кровлями
    layers = markers_to_layers(df_mark)
    print(f"\n  Пластов (между соседними кровлями): {len(layers)}")
    # Нижнюю границу последнего пласта положим по максимальной MD из .dev
    md_max_dev = float(md_dev.max())
    for lay in layers:
        bottom = lay["bottom_md"] if lay["bottom_md"] is not None else md_max_dev
        print(f"    {lay['name']:>15}: {lay['top_md']:.2f} – {bottom:.2f} м")

    # --- 5. Проверки --------------------------------------------------
    print_header("ПРОВЕРКИ")

    all_ok = True

    # 5.1. Отбивки внутри диапазона LAS
    md_mark = df_mark["md"].to_numpy()
    las_min, las_max = float(depth.min()), float(depth.max())
    if md_mark.min() < las_min or md_mark.max() > las_max:
        print(f"  ✗ Некоторые отбивки за пределами LAS "
              f"({las_min:.1f} – {las_max:.1f} м)")
        all_ok = False
    else:
        print(f"  ✓ Все отбивки внутри диапазона LAS")

    # 5.2. Отбивки внутри диапазона, где SP не NULL
    if "SP" in df_las.columns:
        sp_values = df_las["SP"].to_numpy()
        mask_sp = ~np.isnan(sp_values)
        if mask_sp.any():
            sp_lo, sp_hi = mask_span(depth, mask_sp)
            if md_mark.min() < sp_lo or md_mark.max() > sp_hi:
                print(f"  ✗ Не все отбивки попадают в интервал с SP "
                      f"({sp_lo:.1f} – {sp_hi:.1f} м)")
                all_ok = False
            else:
                print(f"  ✓ Все отбивки внутри интервала с SP "
                      f"({sp_lo:.1f} – {sp_hi:.1f} м)")
        else:
            print(f"  ✗ Кривая SP полностью NULL — типизация невозможна")
            all_ok = False

    # 5.3. Инклинометрия покрывает весь диапазон отбивок
    if md_mark.max() > md_dev.max():
        print(f"  ✗ Отбивки уходят глубже последней точки .dev "
              f"({md_mark.max():.1f} > {md_dev.max():.1f})")
        all_ok = False
    else:
        print(f"  ✓ Инклинометрия покрывает все отбивки")

    # 5.4. KB из шапки .dev
    kb = df_dev.attrs.get("kb")
    if kb is not None:
        print(f"  ✓ KB определён: {kb} м")

    # --- Итог --------------------------------------------------------
    print_header("ИТОГ")
    if all_ok:
        print("  Все проверки пройдены. Можно переходить к расчётам.")
    else:
        print("  Есть предупреждения. См. отметки ✗ выше.")


# Стандартный «вход» Python в программу.
# Этот блок выполняется только если файл запущен как скрипт,
# а не импортирован как модуль.
if __name__ == "__main__":
    # np нужен внутри main() для isnan — импортируем на уровне файла
    import numpy as np
    main()
    
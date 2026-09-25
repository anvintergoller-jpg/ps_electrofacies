"""
Чтение инклинометрии в формате Petrel .dev.

Формат .dev:
    - Строки-комментарии начинаются с '#'
    - После комментариев идёт строка-заголовок со столбцами
      (MD X Y Z TVD DX DY AZIM_TN INCL DLS AZIM_GN)
    - Далее — числовые данные, разделённые пробелами/табами
    - Пустых строк может быть много

Возвращает pandas.DataFrame со всеми столбцами из файла.
Плюс извлекает KB (Kelly Bushing) из шапки и прикладывает как атрибут.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_deviation(path, config):
    """
    Читает .dev и возвращает DataFrame.

    Параметры:
        path   : str/Path — путь к .dev
        config : dict — секция deviation из config.yaml
                 (используется только comment_char и key_columns)

    Возвращает:
        pd.DataFrame с числовыми столбцами (MD, X, Y, Z, TVD, ...).

    Побочный эффект:
        Печатает KB из шапки (для информации).
    """
    comment_char = config.get("comment_char", "#")

    header_line = None         # первая «не-комментарийная» строка
    data_rows = []             # список списков чисел
    kb = None                  # Kelly Bushing (из шапки)

    # errors="replace" — если попадётся нечитаемая кодировка,
    # не упадём, а заменим «битые» байты на «?». Для англ. шапки безопасно.
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            # Отрезаем пробелы и перевод строки по краям
            line = raw_line.strip()

            # Пустая строка — пропускаем
            if not line:
                continue

            # Комментарий (обычно шапка) — пропускаем,
            # но ловим в нём KB, если он есть
            if line.startswith(comment_char):
                # Ищем "# WELL DATUM (KB, ...): 113.8 (m)"
                m = re.search(r"WELL DATUM.*?:\s*([-\d.]+)", line)
                if m:
                    try:
                        kb = float(m.group(1))
                    except ValueError:
                        pass
                continue

            # Первая непустая не-комментарийная строка — заголовок
            if header_line is None:
                header_line = line
                continue

            # Всё остальное — данные
            data_rows.append(line.split())

    # Если заголовок так и не нашли — файл битый
    if header_line is None:
        raise ValueError(f"Не найден заголовок столбцов в файле: {path}")

    # Имена столбцов — из заголовка, по пробелам
    columns = header_line.split()

    # Парсим данные: каждая строка должна иметь len(columns) элементов.
    # Если строк с другим числом — пропускаем со счётчиком.
    parsed_rows = []
    skipped = 0
    for parts in data_rows:
        if len(parts) != len(columns):
            skipped += 1
            continue
        try:
            parsed_rows.append([float(x) for x in parts])
        except ValueError:
            # Не числовая строка — пропускаем
            skipped += 1

    if not parsed_rows:
        raise ValueError(f"В файле {path} нет валидных строк данных")

    df = pd.DataFrame(parsed_rows, columns=columns)

    # Прикладываем KB как атрибут DataFrame — pandas это позволяет.
    # Достать потом: df.attrs["kb"]
    df.attrs["kb"] = kb
    df.attrs["source_file"] = str(path)

    if kb is not None:
        print(f"  KB (из шапки .dev): {kb} м")
    if skipped:
        print(f"  [!] Пропущено строк с неверным форматом: {skipped}")

    return df


def md_to_tvd(df_dev, md_values):
    """
    Пересчёт MD → TVD по таблице инклинометрии с линейной интерполяцией.

    Параметры:
        df_dev    : DataFrame из read_deviation
        md_values : array-like — значения MD, для которых нужны TVD
                    (например, глубины из LAS или отбивки)

    Возвращает:
        np.ndarray той же длины, что md_values — соответствующие TVD.

    Как работает:
        В .dev таблица MD→TVD задана на неравномерной сетке
        (шаг около 20 м). Для произвольных MD используем линейную
        интерполяцию np.interp.

    Ограничение:
        Если md_values выходят за диапазон df_dev["MD"] — возвращаются
        крайние TVD (константа). Это осознанный компромисс: лучше
        константа, чем NaN, чтобы не сломать дальнейшие расчёты.
    """
    md_table = df_dev["MD"].to_numpy(dtype=float)
    tvd_table = df_dev["TVD"].to_numpy(dtype=float)

    # np.interp требует, чтобы md_table был отсортирован по возрастанию.
    # Petrel обычно уже отдаёт отсортированным, но перестрахуемся.
    order = np.argsort(md_table)
    md_table = md_table[order]
    tvd_table = tvd_table[order]

    return np.interp(md_values, md_table, tvd_table)
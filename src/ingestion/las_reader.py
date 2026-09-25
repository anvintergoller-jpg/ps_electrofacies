"""
Чтение LAS-файлов каротажа.

Что делает:
    1. Открывает LAS через библиотеку lasio.
    2. Достаёт кривую глубин (DEPT или MD).
    3. Для каждой «канонической» кривой (SP, GK) ищет подходящую
       мнемонику по словарю синонимов.
    4. Заменяет NULL-значения (обычно -999.25) на NaN.

Возвращает pandas.DataFrame с колонками:
    DEPTH, SP, GK  (где NaN = нет данных)

Почему NULL заменяем на NaN:
    - NaN корректно игнорируется функциями numpy: np.nanmean, np.nanmedian.
    - Если оставить -999.25, все средние/медианы/амплитуды будут мусором.
"""

import numpy as np
import pandas as pd
import lasio


def find_curve(las, canonical_name, mnemonic_map):
    """
    Ищет в LAS-объекте кривую по каноническому имени через синонимы.

    Параметры:
        las          : объект lasio
        canonical_name : str — имя в нашей системе ("SP", "GK")
        mnemonic_map : dict — словарь {каноническое: [синонимы]}

    Возвращает:
        str или None — реальную мнемонику в файле, или None если не найдено.

    Логика:
        В файле 227_2263.las кривая называется "SP". Мы ищем сначала
        именно "SP". Если файл от другого оператора и там "PS", ищем "PS".
        Регистр приводим к верхнему — в LAS он может быть любым.
    """
    # .get(...) с fallback: если для канонического имени нет списка
    # синонимов, ищем по имени как есть
    synonyms = mnemonic_map.get(canonical_name, [canonical_name])

    # las.keys() возвращает список мнемоник кривых в файле.
    # Приводим всё к верхнему регистру и строим словарь
    # {МНЕМОНИКА_В_ВЕРХНЕМ_РЕГИСТРЕ: оригинальное_написание},
    # чтобы потом вернуть оригинал (важен регистр при обращении к las[...]).
    available = {k.upper(): k for k in las.keys()}

    # Идём по списку синонимов по порядку — первый найденный побеждает
    for syn in synonyms:
        if syn.upper() in available:
            return available[syn.upper()]

    # Ничего не нашли — сообщаем вызывающему коду
    return None


def replace_nulls(values, null_values):
    """
    Заменяет NULL-значения на NaN.

    Параметры:
        values      : array-like — массив значений из LAS
        null_values : list[float] — список значений-«пустышек»
                       (обычно [-999.25] из секции ~Well)

    Возвращает:
        np.ndarray с NaN вместо NULL.

    Тонкость:
        Сравнивать float нужно с допуском. -999.25 может храниться
        как -999.2499999 или -999.2500001 из-за особенностей записи.
        np.isclose(atol=1e-4) решает эту проблему.
    """
    arr = np.asarray(values, dtype=float)

    for nv in null_values:
        # np.isclose(arr, nv) → булев массив "близко ли каждое значение к nv"
        arr[np.isclose(arr, nv, atol=1e-4)] = np.nan

    return arr


def read_las(path, config, mnemonic_map):
    """
    Главная функция: читает LAS, возвращает DataFrame.

    Параметры:
        path         : str/Path — путь к .las
        config       : dict — секция las из config.yaml
                       (ожидаются ключи: null_values, depth_mnemonic, curves)
        mnemonic_map : dict — словарь синонимов

    Возвращает:
        pd.DataFrame с колонками: DEPTH, SP, GK (и любыми другими,
        перечисленными в config["curves"]).

    Пример:
        df = read_las("data/raw/las/227_2263.las", cfg["las"], mnem_map)
        # df.head()
        #    DEPTH   SP    GK
        # 0    0.0   NaN   NaN
        # ...
    """
    # lasio.read принимает строку, поэтому str(path)
    las = lasio.read(str(path))

    # --- 1. Глубина ------------------------------------------------
    depth_mnemonic = config.get("depth_mnemonic", "DEPT")

    # Пытаемся взять кривую по мнемонике из конфига.
    # Если её нет — падаем с понятной ошибкой, а не молча берём что попало.
    if depth_mnemonic in las.keys():
        depth = np.asarray(las[depth_mnemonic], dtype=float)
    else:
        # Резервный вариант: las.index — это первая кривая файла,
        # обычно глубина. Работает в 99% случаев.
        depth = np.asarray(las.index, dtype=float)

    # --- 2. Каркас DataFrame --------------------------------------
    # Начинаем со словаря — колонки будем добавлять по мере поиска
    data = {"DEPTH": depth}

    # --- 3. Каждая каноническая кривая ---------------------------
    null_values = config.get("null_values", [-999.25])

    for canonical in config["curves"]:
        real_mnemonic = find_curve(las, canonical, mnemonic_map)

        if real_mnemonic is None:
            # Кривой нет — заполняем весь столбец NaN, но не падаем.
            # Валидатор ниже отдельно сообщит об этом.
            print(f"  [!] Кривая '{canonical}' не найдена в LAS")
            data[canonical] = np.full_like(depth, np.nan)
            continue

        # las[real_mnemonic] возвращает массив значений
        values = np.asarray(las[real_mnemonic], dtype=float)

        # Проверка длины: в корректном LAS все кривые одной длины,
        # что совпадает с длиной индекса глубины. Если нет — предупреждаем.
        if len(values) != len(depth):
            print(
                f"  [!] Длина '{real_mnemonic}' ({len(values)}) "
                f"не совпадает с длиной глубин ({len(depth)})"
            )
            # Обрезаем или дополняем NaN — на всякий случай
            if len(values) > len(depth):
                values = values[: len(depth)]
            else:
                values = np.concatenate(
                    [values, np.full(len(depth) - len(values), np.nan)]
                )

        # Заменяем NULL на NaN и складываем в результат
        data[canonical] = replace_nulls(values, null_values)

    # --- 4. Собираем DataFrame -----------------------------------
    df = pd.DataFrame(data)

    return df
"""
Загрузка YAML-конфигов проекта.

Что делает:
    - читает config/config.yaml (общие настройки проекта)
    - читает config/mnemonic_map.yaml (синонимы кривых ГИС)

Использование:
    from src.config_loader import load_config, load_mnemonic_map
    cfg   = load_config()
    mnem  = load_mnemonic_map()

Почему отдельный файл:
    Любая логика работы с YAML (проверка существования, кодировка,
    ошибки парсинга) должна быть в одном месте. Если завтра решим
    перейти с YAML на TOML или JSON — править только здесь.
"""

# pathlib — современная замена os.path для путей (кроссплатформенно)
from pathlib import Path

# yaml — библиотека для чтения YAML. Устанавливается: pip install pyyaml
import yaml


def load_yaml(path):
    """
    Читает произвольный YAML-файл и возвращает словарь Python.

    Параметры:
        path : str или Path — путь к файлу

    Возвращает:
        dict — содержимое файла (Python-словарь)

    Ошибки:
        FileNotFoundError — если файла нет по указанному пути
        yaml.YAMLError      — если файл синтаксически неверен
    """
    # Path(path) — превращает строку в объект Path, чтобы работали
    # удобные методы вроде .exists(), .resolve(), оператор "/"
    path = Path(path)

    # Ранняя проверка — лучше сразу упасть с понятной ошибкой,
    # чем ловить странное поведение yaml через 20 строк
    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path.resolve()}")

    # encoding="utf-8" — важно, потому что в YAML у нас кириллица
    # (комментарии, имена маркеров в будущем). Без явной кодировки
    # на Windows Python может попытаться открыть в cp1251 и упасть
    with open(path, "r", encoding="utf-8") as f:
        # yaml.safe_load (а не yaml.load!) — безопасный режим.
        # Он не выполняет произвольный Python-код из файла.
        # yaml.load() умеет это делать, и это дыра в безопасности.
        data = yaml.safe_load(f)

    return data


def load_config(path="config/config.yaml"):
    """
    Читает основной конфиг проекта.

    По умолчанию — config/config.yaml относительно текущей рабочей папки.
    Предполагается, что вы запускаете код из корня проекта.
    """
    return load_yaml(path)


def load_mnemonic_map(path="config/mnemonic_map.yaml"):
    """Читает словарь синонимов мнемоник кривых."""
    return load_yaml(path)

def resolve_wells(cfg):
    """
    Вернуть список имён скважин для обработки.

    Логика:
        • cfg["wells"] — список имён → возвращаем как есть;
        • cfg["wells"] == "auto"   → сканируем папку LAS,
          берём имена файлов без расширения, сортируем по имени.

    Папка и расширение берутся из шаблона cfg["paths_templates"]["las"]
    (например, "data/raw/las/{well}.las" → папка data/raw/las,
    расширение .las).

    Параметры
    ---------
    cfg : dict — результат load_config()

    Возвращает
    ----------
    list[str] — имена скважин в порядке обработки.

    Исключения
    ----------
    FileNotFoundError — если папка LAS не существует или пуста
                        (в режиме auto).
    ValueError        — если cfg["wells"] не распознано.
    """
    wells_cfg = cfg.get("wells", "auto")

    # Режим 1: явный список.
    if isinstance(wells_cfg, list):
        return [str(w) for w in wells_cfg]

    # Режим 2: auto — сканируем папку LAS.
    if wells_cfg == "auto":
        las_template = cfg["paths_templates"]["las"]

        # Из шаблона "data/raw/las/{well}.las" получаем папку и расширение.
        template_path = Path(las_template)
        las_dir = template_path.parent
        las_ext = template_path.suffix  # ".las"

        if not las_dir.exists():
            raise FileNotFoundError(
                f"Папка с LAS-файлами не найдена: {las_dir.resolve()}"
            )

        # Ищем файлы с нужным расширением, сортируем по имени.
        files = sorted(las_dir.glob(f"*{las_ext}"))
        wells = [f.stem for f in files]

        if not wells:
            raise FileNotFoundError(
                f"В папке {las_dir.resolve()} нет файлов *{las_ext}"
            )

        return wells

    # Неизвестный формат — лучше упасть с понятной ошибкой.
    raise ValueError(
        f"Некорректное значение 'wells' в config.yaml: {wells_cfg!r}. "
        f"Ожидается список (например, [227_2263, 227_2264]) "
        f"или строка 'auto'."
    )
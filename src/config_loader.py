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
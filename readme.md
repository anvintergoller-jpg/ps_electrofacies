# ps_electrofacies

Программа типизации пластов по кривой ПС. См. `docs/PROJECT.md`.

## Установка

1. Установить Python 3.11+.
2. Установить зависимости:
python -m pip install -r requirements.txt
3. Открыть папку в VS Code.

## Запуск
python -m src.run_pipeline

Или через VS Code: **F5** с конфигурацией из `.vscode/launch.json`.

## Структура

    data/raw/           — исходные данные (LAS, .dev, WellTops)
    data/interim/       — промежуточные таблицы (dataset, features, intervals)
    data/processed/     — итоговые результаты
    config/             — YAML с настройками
    src/                — код
    docs/               — методика, решения, план
    results/plots/      — графики
    logs/               — логи (пока не используются)

## Документация

- `docs/PROJECT.md` — что за проект.
- `docs/FEATURES.md` — какие признаки считаем.
- `docs/CLASSIFICATION_RULES.md` — правила классификации формы.
- `docs/ROADMAP.md` — план по шагам.
- `docs/DECISIONS_LOG.md` — почему сделано так, а не иначе.

## Текущий статус

Пройдено: итерации 0.0, 0.1, шаги A, B, C, D.1, D.2.
В работе: шаг D.3 (классификатор формы).
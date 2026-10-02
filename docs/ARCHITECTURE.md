# Архитектура проекта ps_electrofacies

**Дата:** 2026-09-30
**Статус:** актуально (после ревизии H.5)

Этот документ описывает структуру кода, поток данных
и назначение каждого модуля. Нужен, чтобы:

- быстро понять, что где лежит и зачем;
- видеть, какие модули вызывают друг друга;
- знать, где какой параметр применяется;
- не ломать связанные места при правках.

## 1. Общая логика пайплайна

Проект обрабатывает одну или несколько скважин **батчем**.
Для каждой скважины выполняется один и тот же сценарий:
─────────────────────────────────────────────────────────────┐
│ 1. Чтение входных данных │
│ LAS (SP, GK) → .dev (инклинометрия) → отбивки │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Нормализация кривых │
│ SP → aSP (0 = глина, 1 = песок) │
│ GK → GK_norm (0 = песок, 1 = глина) │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Сборка единого датасета │
│ depth_md, depth_tvdss, SP, aSP, GK, GK_norm, layer_name │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Сегментация по пластам │
│ каждый пласт → интервалы reservoir / non_reservoir │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Классификация формы (для reservoir-интервалов) │
│ bell / funnel / cylinder / trapezoid-middle / │
│ symmetric / v-shape / uncertain │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Размер и положение интервала в контейнере │
│ size_class: small / medium / large │
│ position_in_container: top / middle / bottom │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Признаки пласта (уровень A) + сводная типизация │
│ NTG, литопрофиль, container_type │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 8. Визуализация │
│ планшет (SP + aSP + GK + интервалы) │
└─────────────────────────────────────────────────────────────┘
↓ (после всех скважин)
┌─────────────────────────────────────────────────────────────┐
│ 9. Сводная таблица + сопоставление пластов │
│ summary_all_wells_<ts>.csv + cross_well_*.csv │
└─────────────────────────────────────────────────────────────┘
↓ (отдельный скрипт)
┌─────────────────────────────────────────────────────────────┐
│ 10. Выгрузка для Petrel │
│ LAS с FACIES + Points with Attributes │
└─────────────────────────────────────────────────────────────┘


## 2. Структура папок
ps_electrofacies/
├── config/
│ ├── config.yaml # все параметры проекта
│ └── mnemonic_map.yaml # синонимы мнемоник кривых (SP/PS/SPC...)
├── src/
│ ├── config_loader.py # загрузка YAML, resolve_wells
│ ├── ingestion/ # чтение входных файлов
│ │ ├── las_reader.py # LAS → DataFrame
│ │ ├── deviation_reader.py # .dev → DataFrame, MD → TVD
│ │ └── markers_reader.py # отбивки из Excel
│ ├── preprocessing/
│ │ └── normalization.py # SP → aSP, GK → GK_norm
│ ├── application/
│ │ └── build_dataset.py # сборка единой таблицы
│ ├── domain/ # бизнес-логика
│ │ ├── segmentation.py # reservoir / non_reservoir
│ │ ├── classify_form.py # форма аномалии
│ │ ├── reference_curve.py # эталонные кривые для планшета
│ │ ├── size_class.py # размер и положение
│ │ ├── features.py # признаки пласта (уровень A)
│ │ ├── container_type.py # сводная типизация (уровень C)
│ │ └── cross_well_summary.py # сопоставление пластов
│ ├── visualization/
│ │ ├── well_log.py # планшет по скважине
│ │ └── form_legend.py # легенда форм (PNG)
│ ├── io/
│ │ └── petrel_export.py # выгрузка в Petrel
│ ├── run_pipeline.py # основной батч-скрипт
│ ├── validate_inputs.py # валидация данных
│ └── export_to_petrel.py # скрипт выгрузки в Petrel
├── docs/
│ ├── ARCHITECTURE.md # этот файл
│ ├── FEATURES.md # методика признаков
│ ├── CLASSIFICATION_RULES.md # методика классификации формы
│ ├── DECISIONS_LOG.md # журнал решений (ADR)
│ ├── PROJECT_STATE.md # текущее состояние проекта
│ └── ROADMAP.md # план работ
└── data/
├── raw/ # входные файлы
│ ├── las/
│ ├── deviation/
│ └── markers/
└── processed/ # результаты
├── <well_name>/ # по скважине
├── petrel/ # выгрузка для Petrel
├── summary_all_wells_.csv
├── cross_well_.csv
└── run_log_*.txt


## 3. Карта модулей

### 3.1. Конфиг и утилиты

| Модуль | Что делает |
|---|---|
| `config_loader.py` | Читает YAML. Функция `resolve_wells()` — список скважин (auto или явный). |

### 3.2. Ingestion (чтение входных данных)

| Модуль | Что делает |
|---|---|
| `las_reader.py` | Читает LAS через lasio, ищет SP и GK по синонимам, заменяет NULL на NaN. |
| `deviation_reader.py` | Читает .dev (Petrel), достаёт KB. Функция `md_to_tvd()` — интерполяция. |
| `markers_reader.py` | Читает Excel с отбивками, фильтрует по скважине. Функция `markers_to_layers()` — маркеры → пласты. |

### 3.3. Preprocessing (подготовка данных)

| Модуль | Что делает |
|---|---|
| `normalization.py` | `normalize_curves()`: SP → aSP (0 = глина), GK → GK_norm. Перцентили 2/98 внутри рабочего интервала. |

### 3.4. Application (сборка датасета)

| Модуль | Что делает |
|---|---|
| `build_dataset.py` | Соединяет LAS + .dev + отбивки в одну таблицу. Размечает точки по пластам (по MD). |

### 3.5. Domain (бизнес-логика)

| Модуль | Что делает |
|---|---|
| `segmentation.py` | Сегментирует каждый пласт на интервалы reservoir / non_reservoir по порогу `reservoir.cutoff`. Функция `smooth()` используется всеми остальными модулями для сглаживания aSP. |
| `classify_form.py` | Классифицирует форму каждого reservoir-интервала. 7 типов. |
| `reference_curve.py` | Строит эталонную кривую aSP для каждого типа формы (для наложения на планшет). |
| `size_class.py` | Определяет размер интервала (доля в контейнере) и положение центроида (top/middle/bottom). |
| `features.py` | Считает признаки уровня A: статистика точек пласта + литологический профиль. |
| `container_type.py` | Собирает сводную типизацию пласта: NTG, доминирующая форма/размер/положение, container_type. |
| `cross_well_summary.py` | Сопоставляет одноимённые пласты между скважинами: матрицы, счётчики, нетипичные. |

### 3.6. Visualization

| Модуль | Что делает |
|---|---|
| `well_log.py` | Рисует 4-панельный планшет: глубины, SP+aSP, GK+GK_norm, колонка интервалов. Накладывает эталонные кривые по формам. |
| `form_legend.py` | Рисует справочный PNG с типами форм. Палитра `FORM_COLORS` — общая с `well_log.py`. |

### 3.7. IO

| Модуль | Что делает |
|---|---|
| `petrel_export.py` | Экспортирует LAS с FACIES и Points with Attributes для Petrel. Транслитерация русских имён пластов. |

### 3.8. Скрипты (точки входа)

| Скрипт | Что запускает |
|---|---|
| `run_pipeline.py` | Полный батч-цикл по всем скважинам. Создаёт `summary_all_wells`, планшеты, `form_legend`, `cross_well_*`. |
| `validate_inputs.py` | Проверка входных данных по всем скважинам (без обработки). |
| `export_to_petrel.py` | Выгрузка готовых результатов в Petrel. |

## 4. Поток данных (детально)

### 4.1. Обработка одной скважины (`process_well`)
LAS-файл .dev-файл Excel (общий)
│ │ │
▼ ▼ ▼
read_las() read_deviation() read_markers()
│ │ │
│ df_las: DEPTH, SP, GK │ df_dev: MD, TVD... │ df_mark
│ │ │
└──────────────┬─────────────┘ │
│ │
▼ │
Определение рабочего интервала │
top_md = min(md отбивок) │
bottom_md = max(MD из .dev) │
│ │
▼ │
normalize_curves(df_las, ...) │
→ aSP, GK_norm │
│ │
└──────────────┬──────────────────────┘
│
▼
build_dataset(df_las, df_dev, df_mark)
│
│ df: depth_md, depth_tvdss,
│ SP, aSP, GK, GK_norm, layer_name
│ df.attrs: layers, kb
▼
segment_all_layers(df, layers, ...)
│
│ intervals_df: layer_name, kind,
│ top_md, top_tvdss...
▼
classify_intervals(df, intervals_df, ...)
│
│ + form_type, confidence, sp_top...
▼
classify_size_and_position(...)
│
│ + size_class, position_in_container
▼
compute_container_features(df, layers)
│
│ + признаки уровня A (n_points,
│ sp_coverage, литопрофиль)
▼
compute_container_types(features, intervals, ...)
│
│ + NTG, dominant_*, container_type
▼
plot_well(df, ...) → PNG


### 4.2. Сборка и вывод (в `main`)

После цикла по скважинам:
features_df по каждой скважине
│
▼
pd.concat → summary_df (одна строка на пласт × скважину)
│
├─→ summary_all_wells_<ts>.csv
│
▼
build_cross_well_summary()
│
├─→ cross_well_matrix.csv
├─→ cross_well_matrix_ntg_tvdss.csv
├─→ cross_well_counts.csv
└─→ cross_well_atypical.csv

run_log_lines → run_log_<ts>.txt
legend → form_legend.png


## 5. Параметры config — где применяются

Ключевые параметры и модули, которые их используют.

| Параметр (config) | Модули |
|---|---|
| `wells`, `paths_templates` | `config_loader.resolve_wells`, `run_pipeline`, `validate_inputs`, `export_to_petrel` |
| `las.*` | `las_reader` |
| `deviation.*` | `deviation_reader` |
| `markers.*` | `markers_reader` |
| `normalization.*` | `normalization` |
| `preprocessing.smooth_window` | `segmentation`, `classify_form`, `features`, `cross_well` |
| `reservoir.cutoff` | `segmentation`, `classify_form`, `well_log`, `form_legend` |
| `reservoir.min_thickness_tvdss` | `segmentation` |
| `classification.*` | `classify_form` |
| `size_classification.*` | `size_class` |
| `container_type.dominance_margin` | `container_type` |
| `cross_well_summary.numeric_metric` | `cross_well_summary` |
| `petrel_export.*` | `petrel_export`, `export_to_petrel` |
| `output.*` | `run_pipeline`, `export_to_petrel` |

**Принцип:** один параметр — одно место в config. В коде — никаких
хардкодов значений; всё читается из `cfg[...]` и передаётся
в функции **явными аргументами**.

## 6. Соглашения по коду

- **Наименование шкалы.** `aSP` — приведённый ПС (0 = глина, 1 = песок).
  `GK_norm` — нормированный ГК (0 = песок, 1 = глина).
  Не путать: шкалы **противоположны**.
- **Переменные.** `sp_top`, `sp_mid`, `sp_bot` — опорные точки формы
  (в шкале aSP). Несмотря на имя `sp_*`, значения — это aSP.
- **Docstrings.** Каждая публичная функция — с описанием параметров
  и возвращаемого значения.
- **Ошибки.** Модуль не должен «угадывать» параметры. Если
  обязательный параметр не задан — `ValueError`, а не дефолт.
- **Логирование.** Внутренние модули не пишут в файлы. `run_pipeline`
  перехватывает `print` и сохраняет в `<well>/log.txt`.

## 7. Что изменено в ревизии H.5

- Введены секции `preprocessing` и `reservoir` в config — общие
  параметры, которые раньше дублировались.
- Убрана секция `segmentation` — её параметры переехали в общие.
- Убрана секция `logging` (не использовалась).
- Убраны legacy-пути `interim_dir`, `results_dir`, `logs_dir`.
- Все хардкоды `0.4` и `5` в модулях заменены на чтение из config.
- Убран скрытый дефолт `reservoir_cutoff=0.60` в `ClassificationParams`.
- Исправлен баг в `_has_multiple_minima` (`<` → `>` для шкалы aSP).


# Forecasting Customer Churn

Проект для бинарной классификации оттока клиентов телеком-сервиса на датасете `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

Основная цель: предсказать, уйдёт клиент (`Churn=Yes`) или останется (`Churn=No`), и получить воспроизводимый пайплайн от EDA до обучения нейросети.

## Что сделано в проекте

- Базовый EDA + проверка качества данных.
- Подготовка признаков (numeric/binary/categorical).
- Обучение MLP-модели на PyTorch.
- Сохранение артефактов: отчёт, графики, веса модели, препроцессор.

## Стек

- `Python 3.10+`
- `pandas`, `numpy`
- `scikit-learn`
- `torch`
- `matplotlib`

## Структура репозитория

```text
forecasting_customer_churn/
├─ data/
│  ├─ raw/
│  │  └─ WA_Fn-UseC_-Telco-Customer-Churn.csv   # исходный датасет
│  └─ processed/                                # подготовленные выборки (пока пусто)
│
├─ notebooks/                                   # Jupyter-ноутбуки (сейчас пусто)
│
├─ src/
│  ├─ data/
│  │  ├─ __init__.py
│  │  └─ eda.py                                 # EDA: проверки качества + графики + markdown-отчёт
│  ├─ features/
│  │  ├─ __init__.py
│  │  └─ preprocessing.py                       # очистка данных, split, DataLoader, ColumnTransformer
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ mlp.py                                 # архитектура ChurnMLP
│  └─ train/
│     ├─ __init__.py
│     └─ train.py                               # обучение, валидация, early stopping, сохранение модели
│
├─ artifacts/
│  ├─ figures/
│  │  └─ eda/
│  │     ├─ churn_balance.png
│  │     ├─ churn_rate_by_contract.png
│  │     └─ monthly_charges_distribution.png
│  ├─ metrics/                                  # метрики/логи экспериментов (пока пусто)
│  ├─ models/                                   # .pt и .pkl после обучения
│  └─ reports/
│     └─ eda_telco_churn_report.md
│
├─ tests/
│  ├─ unit/                                     # unit-тесты (каркас)
│  └─ integration/                              # интеграционные тесты (каркас)
│
└─ README.md
```

## Что где лежит и зачем

- `src/data/eda.py`
  - Загружает и валидирует данные.
  - Проверяет пропуски, дубликаты, логические несоответствия.
  - Строит графики EDA и сохраняет их в `artifacts/figures/eda/`.
  - Генерирует markdown-отчёт в `artifacts/reports/`.

- `src/features/preprocessing.py`
  - Очистка данных (`load_and_clean`).
  - Кодирование бинарных полей и таргета.
  - Построение `ColumnTransformer`:
    - numeric -> `StandardScaler`
    - binary -> `passthrough`
    - categorical -> `OneHotEncoder`
  - Разбиение train/val/test + формирование `DataLoader`.

- `src/models/mlp.py`
  - MLP с `BatchNorm` и `Dropout` для бинарной классификации.

- `src/train/train.py`
  - Полный train-пайплайн:
    - подготовка данных
    - обучение по эпохам
    - early stopping по `val AUC`
    - финальная оценка на test (`ROC-AUC`, `F1`, `classification_report`)
    - сохранение модели и препроцессора в `artifacts/models/`.

## Как запускать

Из корня проекта:

```bash
python -m src.data.eda
python -m src.train.train
```

Если зависимости ещё не установлены:

```bash
pip install pandas numpy scikit-learn matplotlib torch
```

## Как должно быть организовано дальше (рекомендуемый стандарт)

- `data/raw/`:
  - только неизменённые исходные файлы.
- `data/processed/`:
  - очищенные/преобразованные таблицы и фичи для экспериментов.
- `artifacts/models/`:
  - только сериализованные артефакты обучения (`.pt`, `.pkl`).
- `artifacts/metrics/`:
  - экспорт метрик по запускам (например, `metrics_YYYYMMDD.json`).
- `tests/unit/`:
  - тесты отдельных функций (`load_and_clean`, `build_preprocessor`, `prepare_loaders`).
- `tests/integration/`:
  - тесты полного пайплайна (`EDA` и `train` на небольшом сэмпле).

## Минимальный рабочий сценарий проекта

1. Положить исходный CSV в `data/raw/`.
2. Запустить `python -m src.data.eda` и проверить отчёт/графики в `artifacts/`.
3. Запустить `python -m src.train.train`.
4. Проверить, что появились:
   - `artifacts/models/churn_mlp.pt`
   - `artifacts/models/preprocessor.pkl`
5. (Следующий шаг) добавить тесты в `tests/unit` и `tests/integration`.

# ChemAI: Predict the Cure

Воспроизводимый ML-пайплайн для предсказания `IC50`, `CC50` и `SI` по признакам химических соединений.

## Данные

Ожидаемые файлы:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

В текущей версии датасета:
- `train.csv`: 751 строка, 214 колонок;
- `test.csv`: 250 строк, 211 колонок;
- targets в train: `IC50, mM`, `CC50, mM`, `SI`;
- submission содержит: `index`, `IC50`, `CC50`, `SI`;
- SMILES-колонки нет, поэтому базовый пайплайн использует уже рассчитанные числовые RDKit-like descriptors и fragment counts.

## Методы

Проверяются:
- DummyRegressor как нижняя граница;
- Ridge как линейный baseline;
- RandomForest / ExtraTrees / HistGradientBoosting;
- optional LightGBM / CatBoost / XGBoost;
- отдельные модели для каждого target;
- стратегии SI: прямой прогноз, расчет `CC50 / IC50`, blend.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Опционально:

```bash
pip install lightgbm catboost xgboost optuna rdkit-pypi
```

## Запуск

```bash
python run_pipeline.py --config configs/config.yaml --stage all
```

Submission появится в:

```text
data/submissions/baseline_rdkit_descriptors_submission.csv
```

## Воспроизводимость

- все seed заданы в `configs/config.yaml`;
- данные не меняются в процессе обучения;
- метрики, OOF и модели сохраняются в `models/<experiment_name>/`;
- для выбора решения используется cross-validation, а не public leaderboard.

## Проверки

В пайплайне и EDA нужно зафиксировать:
- пропуски;
- константные признаки;
- дубликаты;
- выбросы target;
- согласованность `SI = CC50 / IC50`;
- отсутствие target leakage;
- сравнение прямого прогноза SI с расчетом через отношение;
- сравнение моделей по mean RMSE на CV.

## Исключение методов

Метод остается в финальном решении только если:
1. улучшает CV RMSE;
2. повышает надежность или воспроизводимость;
3. не создает leakage;
4. не усложняет решение непропорционально приросту качества.

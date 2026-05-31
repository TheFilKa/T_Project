# ChemAI: Predict the Cure

Воспроизводимый ML-пайплайн для задачи ChemAI: предсказание трех показателей биологической активности химических соединений:

- `IC50` — концентрация, подавляющая 50% активности вируса;
- `CC50` — концентрация, токсичная для 50% клеток;
- `SI` — Selectivity Index.

Финальная оценка считается через RMSE, поэтому все эксперименты сравниваются по mean CV RMSE и дополнительно проверяются на Kaggle submission.

## Текущий лучший Kaggle-tested config

Лучший проверенный на Kaggle вариант на момент фиксации репозитория:

```yaml
project:
  seed: 72

data:
  drop_cols: ["index"]

training:
  model_name: "random_forest"
  target_transform: "none"
  si_strategy: "ratio"
  ic50_floor_for_si: 1.0e-9

models:
  random_forest:
    n_estimators: 1000
    max_depth: null
    min_samples_split: 2
    min_samples_leaf: 2
    max_features: 0.75
    bootstrap: true
    random_state: 72
    n_jobs: -1
```

Лучший Kaggle score для этой ветки: `277.85392`.

## Данные

Файлы соревнования должны быть размещены локально:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

Эти файлы не коммитятся в GitHub. Директория `data/raw/` содержит только `.gitkeep`.

В использованной версии данных:

- `train.csv`: 751 строка, 214 колонок;
- `test.csv`: 250 строк, 211 колонок;
- targets в train: `IC50, mM`, `CC50, mM`, `SI`;
- submission содержит: `index`, `IC50`, `CC50`, `SI`;
- SMILES-колонки нет, поэтому базовый пайплайн использует уже рассчитанные числовые химические дескрипторы и fragment-count признаки.

## Установка

### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Если установка в `.venv` недоступна из-за proxy, можно запускать проект глобальным Python, если зависимости уже установлены:

```powershell
python -c "import pandas, numpy, sklearn, yaml, joblib, scipy; print('OK')"
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Запуск

```bash
python run_pipeline.py --config configs/config.yaml --stage all
```

После запуска создаются:

```text
models/baseline_rdkit_descriptors/metrics.json
models/baseline_rdkit_descriptors/oof_predictions.csv
models/baseline_rdkit_descriptors/models.joblib
data/submissions/baseline_rdkit_descriptors_submission.csv
```

Эти артефакты локальные и не коммитятся.

## Структура проекта

```text
.
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── submissions/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── models.py
│   ├── predict.py
│   ├── train.py
│   ├── utils.py
│   └── validation.py
├── average_submissions.py
├── requirements.txt
├── run_pipeline.py
└── README.md
```

## Воспроизводимость

- Все основные параметры лежат в `configs/config.yaml`.
- Для лучшего Kaggle-tested решения зафиксированы `project.seed=72` и `random_forest.random_state=72`.
- `index` исключен из признаков через `drop_cols`.
- Train/test targets не смешиваются.
- Test targets не используются.
- Submission создается только через обученные модели и `sample_submission.csv`.
- Модели, OOF-предсказания и метрики сохраняются локально.

## Метрика

Для каждого target считается RMSE на OOF-предсказаниях. Итоговая CV-метрика:

```text
mean_rmse = mean(RMSE_IC50, RMSE_CC50, RMSE_SI)
```

## Что было проверено

### 1. `SI` как отдельный target или derived target

В train почти точно выполняется:

```text
SI = CC50 / IC50
```

Поэтому были проверены три стратегии:

- `direct`: отдельная модель для `SI`;
- `ratio`: `SI = pred_CC50 / pred_IC50`;
- `blend`: смесь прямого прогноза SI и ratio-SI.

Вывод:

- `direct` дал более слабый CV на раннем baseline;
- `ratio` оказался лучшей Kaggle-tested стратегией;
- `blend` улучшал KFold CV, но текущий лучший Kaggle submission использует точное ratio, поэтому в финальном config оставлен `ratio`.

### 2. Удаление `index`

`index` не является химическим признаком и может приводить к переобучению на порядок строк.

Проверка показала, что исключение `index` улучшило CV и сделало решение более корректным, поэтому используется:

```yaml
data:
  drop_cols: ["index"]
```

### 3. Target transform

Проверялись:

- `target_transform: "log1p"`;
- `target_transform: "none"`.

Несмотря на скошенные targets и выбросы, обучение на исходной шкале лучше согласуется с RMSE на исходной шкале. В финальном config используется:

```yaml
training:
  target_transform: "none"
```

### 4. RandomForest tuning

Проверялись параметры:

- `min_samples_leaf`: 1, 2, 3;
- `max_features`: 0.6, 0.65, 0.7, 0.725, 0.75, 0.8, `sqrt`;
- `bootstrap`: true/false;
- `n_estimators`: 1000/1500;
- разные seed: 42, 52, 62, 72, 82.

На Kaggle лучше всего перенесся вариант:

```yaml
seed: 72
min_samples_leaf: 2
max_features: 0.75
bootstrap: true
n_estimators: 1000
```

### 5. Seed averaging

Проверялось усреднение submission нескольких seed (`42 + 62 + 72`). На Kaggle оно оказалось немного хуже одиночного seed 72, поэтому в финальный submission не включено.

## Почему некоторые модели и методы не вошли в финал

### DummyRegressor

Используется только как sanity-check baseline. Не подходит для финала, потому что не использует химические признаки.

### Ridge

Полезен как быстрый линейный baseline, но химические дескрипторы имеют нелинейные связи с target. Ridge не был выбран как финальная модель, потому что основная tree-based модель давала более сильный результат.

### ExtraTrees

Подходящий кандидат для малых табличных данных, но в текущей серии экспериментов Kaggle-tested best был получен на RandomForest. ExtraTrees оставлен в config как воспроизводимая альтернатива для будущих запусков.

### HistGradientBoosting

Быстрый sklearn-бустинг без внешних зависимостей. Не выбран как финальный вариант, потому что текущая стабильная Kaggle-tested ветка построена на RandomForest, а дополнительное усложнение не было оправдано лучшим проверенным результатом.

### LightGBM / CatBoost / XGBoost

Оставлены как optional-подходы. Они могут быть сильны на табличных данных, но добавляют внешние зависимости и риск переобучения на маленьком train (`751` объект). В финальную версию они не включены, потому что best Kaggle-tested результат был достигнут без них.

### `log1p` target transform

Не включен в финал: CV ухудшался относительно обучения на исходной шкале, а Kaggle RMSE считается также на исходной шкале.

### `blend` для SI

`blend` улучшал локальный KFold CV, но лучший Kaggle submission соответствовал строгому расчету `SI = CC50 / IC50`. Поэтому финальная Kaggle-tested версия использует `ratio`.

### Seed averaging

Усреднение нескольких seed не улучшило Kaggle score относительно одиночного seed 72. Метод исключен из финала, но скрипт `average_submissions.py` оставлен для повторной проверки.

## Чек-лист перед Kaggle submission

- [ ] В `data/raw/` лежат `train.csv`, `test.csv`, `sample_submission.csv`.
- [ ] В `configs/config.yaml` стоят seed `72`, `random_state: 72`, `max_features: 0.75`.
- [ ] `training.si_strategy: "ratio"`.
- [ ] `data.drop_cols: ["index"]`.
- [ ] Команда `python run_pipeline.py --config configs/config.yaml --stage all` проходит без ошибок.
- [ ] В submission нет `NaN`, `inf`, отрицательных значений.
- [ ] Submission содержит ровно колонки `index`, `IC50`, `CC50`, `SI`.
- [ ] Не использовались test targets, public leaderboard не использовался для ручной подгонки target-значений.

## Git hygiene

В репозиторий не должны попадать:

- `.venv/`;
- `__pycache__/`;
- `data/raw/*.csv`;
- `data/submissions/*.csv`;
- `models/*.joblib`, OOF и metrics;
- локальные cache/log файлы.

Если такие файлы уже были добавлены в Git, удалить их из индекса можно так:

```bash
git rm --cached -r .venv data/raw data/processed data/submissions models src/__pycache__
git add .gitignore .gitattributes README.md configs/config.yaml src/ run_pipeline.py requirements.txt average_submissions.py
git commit -m "Clean repository and document best RandomForest baseline"
git push origin main
```

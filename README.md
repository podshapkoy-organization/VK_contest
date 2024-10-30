# Решение задачи классификации временных рядов

## Файлы в решении

- `train.parquet`: Обучающая выборка.
- `test.parquet`: Тестовая выборка.
- `sample_submission.csv`: Пример решения.
- `submission.csv`: Файл с предсказаниями.
- `model.pkl`: Обученная модель.
- `script.py`: Скрипт для предсказания.

## Инструкция по запуску

1. Установите необходимые библиотеки:
   ```bash
   pip install pandas matplotlib seaborn statsmodels scikit-learn xgboost joblib
2. Запустите скрипт для предсказания:
    ```bash
    python script.py
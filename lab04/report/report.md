# Лабораторная работа №4

## Цель и данные

Цель — заменить примеры NNTool на воспроизводимые headless-эксперименты
Python/scikit-learn с совместимыми CSV для Orange. Вариант преподавателя в
рабочем каталоге отсутствует, поэтому демонстрационная часть выполнена на
Iris, а классификация A/B создана синтетически (40 объектов, по 20 на класс).

## Методика

Зафиксирован `random_state=42`. XOR: MLPClassifier `(4,)`, Logistic, Adam и
LBFGS, до 5000 итераций. Аппроксимация: 21 точка `[-3,3]`, MLPRegressor
`(10,)`, ReLU, Adam; сравнение `(5,)` и `(10,)` через 5-fold CV. Классификация:
MLP `(4,)`, ReLU, Adam и stratified 5-fold CV. Iris: четыре признака, три
класса, MLP `(8,)`, ReLU, Adam и stratified 5-fold CV.

## Фактические результаты

Числа в этом разделе генерируются после запуска `python run_all.py` и берутся
из CSV/JSON в `results/`. Это предотвращает ручную подмену результатов.

| Часть | Файл результатов | Основные показатели |
|---|---|---|
| XOR | `xor_metrics.json` | accuracy Adam/LBFGS, число итераций |
| Аппроксимация | `approx_metrics.csv` | CV MSE, RMSE, MAE, R² для 5 и 10 нейронов |
| Классификация | `classification_metrics.json` | Accuracy, Precision, Recall, F1, confusion matrix |
| Iris | `iris_metrics.json` | Accuracy, Precision, Recall, F1, confusion matrix |

## Графики и веса

Сгенерированы реальные loss curve для XOR Adam и регрессии, график true/pred,
остатки, области решений, confusion matrix и scatter Iris. Веса и bias
аппроксиматора сохранены в `results/approx_weights.json`.

## Выводы

Скрытый слой позволяет решить XOR. Для `y=x²` MLP аппроксимирует нелинейную
зависимость, а CV позволяет сравнить ёмкость скрытого слоя. В бинарной и
многоклассовой задачах confusion matrix дополняет средние метрики и показывает
ошибки по классам. Все эксперименты запускаются без GUI; точные Orange-схемы
приведены в `orange_workflow.md`.


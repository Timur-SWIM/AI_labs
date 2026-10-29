# Схемы Orange

Все CSV имеют строку заголовка и подходят для виджета **File**. В `File` назначьте
числовые столбцы как Features, а целевой столбец как Target (XOR `result`,
классификация `class`, Iris `class`). Для XOR строки `false/true` являются
дискретным классом.

## XOR

```text
File(data/xor.csv) -> Select Columns -> Neural Network
Select Columns -> Predictions
Neural Network -> Predictions
Select Columns -> Test and Score (Test on train data)
Neural Network -> Test and Score
Predictions -> Data Table
```

Neural Network: hidden layers `4`, activation `Logistic`, solver/optimization
настройки Orange по умолчанию. В Test and Score выберите **Test on train data**.

## Аппроксимация

`File(data/approximation.csv)` (при необходимости экспортируйте
`results/approx_predictions.csv`, выбрав `x` как Feature и `y_true` как Target)
подключается по той же схеме. Для просмотра специальных графиков можно
подключить `Predictions -> Python Script`.

## Классификация и Iris

Для `data/classification.csv` и `data/iris.csv` используется та же цепочка,
а также:

```text
Test and Score -> Confusion Matrix
Predictions -> Python Script
```

Для обеих задач в Test and Score выберите 5-fold cross validation. Для MLP
классификации задайте hidden layers `4`, для Iris — `8`, activation `ReLU`,
а данные предварительно масштабируйте через Normalize при необходимости.

Важно: Predictions должен получить и **Data**, и **Model**. Test and Score
должен получить и **Data**, и **Learner**. Не используйте Selected Predictions
вместо полного Predictions без необходимости.


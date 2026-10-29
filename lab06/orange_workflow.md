# Orange workflow

Поскольку стандартный Orange Neural Network решает задачу с одной целевой
переменной, подготовлены два совместимых файла:

- `data/coord_x.csv`: Features = `rho, phi`, Target = `x`;
- `data/coord_y.csv`: Features = `rho, phi`, Target = `y`.

Для каждого файла соберите схему:

```text
File -> Select Columns -> Neural Network
Select Columns -> Predictions
Neural Network -> Predictions
Select Columns -> Test and Score
Neural Network -> Test and Score
Predictions -> Data Table
```

В `Select Columns` поместите `rho` и `phi` в Features, соответствующий `x`
или `y` — в Target. В Neural Network задайте hidden layers `7, 3`, activation
`tanh`, regression, max iterations `3000`, replicable training. В Test and
Score используйте 5-fold Cross Validation.

Важно: Predictions должен получать одновременно Data и Model; Test and Score
— Data и Learner. Основной двухвыходный эксперимент 2–7–3–2 остаётся в
Python, потому что он совместно оптимизирует `x` и `y`.


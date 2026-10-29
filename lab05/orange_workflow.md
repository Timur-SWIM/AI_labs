# Orange workflow

Orange используется только для просмотра CSV. Основная модель реализована в
Python, чтобы явно показать формулы Hamming и две реализации MAXNET.

## Динамика и эксперименты

```text
File -> Python Script -> Data Table
                     -> Line Plot
                     -> Scatter Plot
```

В `Python Script` загрузите один из файлов `results/noise_accuracy.csv`,
`results/weight_perturbation.csv` или `results/epsilon_sweep.csv` и передайте
его как Data Table. Для `epsilon_sweep.csv` постройте `epsilon` по оси X и
`iterations` по оси Y; для noise — `distance` и accuracy; для весов —
`perturbation` и accuracy.

`data/cross.csv`, `data/square.csv`, `data/diamond.csv` и
`results/exact_results.csv` открываются виджетом File. В последнем числовые
поля — потенциалы и число итераций, а текстовые поля — классы/победители.


from pathlib import Path
import pandas as pd
import json

ROOT=Path(__file__).parents[1]

def test_outputs_exist():
    for name in ['exact_results.csv','noise_trials.csv','noise_accuracy.csv','weight_perturbation.csv','epsilon_sweep.csv','metadata.json']:
        p=ROOT/'results'/name; assert p.exists() and p.stat().st_size>0
    for name in ['reference_patterns.png','accuracy_vs_hamming_distance.png','accuracy_vs_weight_perturbation.png','iterations_vs_epsilon.png']:
        assert (ROOT/'plots'/name).exists()

def test_three_classes_and_eight_epsilons():
    meta=json.loads((ROOT/'results/metadata.json').read_text())
    assert len(meta['classes'])==3
    assert len(pd.read_csv(ROOT/'results/epsilon_sweep.csv'))>=8

def test_exact_classification():
    df=pd.read_csv(ROOT/'results/exact_results.csv')
    expected={'cross':'cross','square':'square','diamond':'diamond'}
    feedforward=df.groupby('input').feedforward_winner.first().to_dict()
    recurrent=df.groupby('input').recurrent_winner.first().to_dict()
    assert feedforward == expected
    assert recurrent == expected


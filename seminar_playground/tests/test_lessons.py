import numpy as np
from src.lessons import neuron_metrics, descent_step

def test_neuron_one_and_two_examples():
    assert neuron_metrics(3,1,False)["sse"] == 0
    assert neuron_metrics(3,1,True)["sse"] == 0
    assert neuron_metrics(1,0,True)["sse"] == 74

def test_descent_exact_step_and_oscillation():
    assert descent_step(-1,.5)["next_w"] == 2
    assert descent_step(-1,1)["next_w"] == 5
    assert descent_step(5,1)["next_w"] == -1

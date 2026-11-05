import numpy as np
from src.training_engine import TrainingEngine
from src.experiments import make_split, evaluate

def data():
    rng=np.random.default_rng(2); x=rng.normal(size=(8,2)); y=np.column_stack((x[:,0]**2+x[:,1],x[:,0]-x[:,1]))
    return x,y

def test_shapes_parameter_count_and_gradient():
    e=TrainingEngine((2,7,3,2)); assert e.parameter_count==53; assert e.predict(np.zeros((5,2))).shape==(5,2)
    x,y=data(); gw,_=e.gradients(x,y); eps=1e-6; old=e.weights[0][0,0]
    e.weights[0][0,0]=old+eps; plus=e.mse(y,e.predict(x)); e.weights[0][0,0]=old-eps; minus=e.mse(y,e.predict(x)); e.weights[0][0,0]=old
    assert np.isclose(gw[0][0,0],(plus-minus)/(2*eps),rtol=1e-4,atol=1e-5)

def test_pause_resume_is_identical_and_seed_reset():
    x,y=data(); first=TrainingEngine((2,5,2),seed=8); second=TrainingEngine((2,5,2),seed=8)
    first.train_epochs(20,x,y); second.train_epochs(7,x,y); second.train_epochs(13,x,y)
    for a,b in zip(first.weights,second.weights): assert np.allclose(a,b)
    for a,b in zip(first.mw,second.mw): assert np.allclose(a,b)
    reset=TrainingEngine((2,5,2),seed=8)
    for a,b in zip(first.snapshot()["weights"], TrainingEngine((2,5,2),seed=8).weights): assert not np.allclose(a,b) or first.epoch == 0
    for a,b in zip(reset.weights, TrainingEngine((2,5,2),seed=8).weights): assert np.array_equal(a,b)

def test_train_normalization_and_session_independence_and_metrics():
    split=make_split(); old=split.mean.copy(); split.control.loc[:,"rho"]+=999
    assert np.array_equal(old,split.mean)
    a,b=TrainingEngine((2,4,2),seed=1),TrainingEngine((2,4,2),seed=1)
    x,y=split.arrays(split.train); a.train_epochs(3,x,y)
    assert not np.allclose(a.weights[0],b.weights[0])
    m=TrainingEngine.metrics([[0,0],[0,0]],[[.01,.02],[.2,0]],.05); assert m["gf"]==1 and m["bf"]==1 and m["mse"]>0

def test_step_engine_matches_original_lab06_formula():
    # Direct independent full-batch Adam implementation for one epoch.
    x,y=data(); e=TrainingEngine((2,3,2),seed=42,learning_rate=.01)
    gw,gb=e.gradients(x,y); before=[w.copy() for w in e.weights]
    expected=[w-.01*((.1*g)/(1-.9))/(np.sqrt((.001*g*g)/(1-.999))+1e-8) for w,g in zip(before,gw)]
    e.train_epoch(x,y)
    for got,want in zip(e.weights,expected): assert np.allclose(got,want)

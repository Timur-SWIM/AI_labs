#!/usr/bin/env python3
"""Run all Lab 04 experiments and regenerate data, metrics and plots."""
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent
DATA, RESULTS, PLOTS = ROOT / "data", ROOT / "results", ROOT / "plots"
SEED, DPI = 42, 180
plt.rcParams.update({"figure.dpi": DPI, "savefig.dpi": DPI, "axes.grid": True})

def savefig(path):
    plt.tight_layout(); plt.savefig(path, dpi=DPI); plt.close()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

def xor():
    x = np.array([[0,0],[0,1],[1,0],[1,1]], float); y = np.array([0,1,1,0])
    pd.DataFrame({"x1": x[:,0].astype(int), "x2": x[:,1].astype(int), "result": ["false","true","true","false"]}).to_csv(DATA/"xor.csv", index=False)
    models = {
        "adam": MLPClassifier(hidden_layer_sizes=(4,), activation="logistic", solver="adam", max_iter=5000, random_state=SEED),
        "lbfgs": MLPClassifier(hidden_layer_sizes=(4,), activation="logistic", solver="lbfgs", max_iter=5000, random_state=SEED),
    }
    info, rows = {}, []
    for name, model in models.items():
        model.fit(x, y); pred = model.predict(x)
        info[name] = {"accuracy": float(accuracy_score(y,pred)), "iterations": int(model.n_iter_), "parameters": model.get_params()}
        for a,b,t,p in zip(x[:,0],x[:,1],y,pred): rows.append({"x1":int(a),"x2":int(b),"target":int(t),"prediction":int(p),"model":name})
        if name == "adam":
            plt.figure(); plt.plot(np.arange(1,len(model.loss_curve_)+1), model.loss_curve_); plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.title("XOR: Adam training curve"); savefig(PLOTS/"xor_training_curve.png")
    pd.DataFrame(rows).to_csv(RESULTS/"xor_predictions.csv", index=False); write_json(RESULTS/"xor_metrics.json", info)

def approximation():
    x = np.linspace(-3,3,21); y=x**2; X=x.reshape(-1,1)
    model = MLPRegressor(hidden_layer_sizes=(10,), activation="relu", solver="adam", max_iter=5000, random_state=SEED)
    model.fit(X,y); pred=model.predict(X); residual=y-pred
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows=[]
    for hidden in (5,10):
        estimator=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(hidden,), activation="relu", solver="adam", max_iter=5000, random_state=SEED))
        scores=cross_validate(estimator,X,y,cv=cv,scoring={"mse":"neg_mean_squared_error","mae":"neg_mean_absolute_error","r2":"r2"})
        rows.append({"hidden_neurons":hidden,"cv_mse_mean":float(-scores["test_mse"].mean()),"cv_rmse_mean":float(np.sqrt(-scores["test_mse"].mean())),"cv_mae_mean":float(-scores["test_mae"].mean()),"cv_r2_mean":float(scores["test_r2"].mean())})
    pd.DataFrame({"x":x,"y_true":y,"y_pred":pred,"residual":residual}).to_csv(RESULTS/"approx_predictions.csv",index=False)
    pd.DataFrame(rows).to_csv(RESULTS/"approx_metrics.csv",index=False)
    write_json(RESULTS/"approx_weights.json", {"hidden_layer_sizes":list(model.hidden_layer_sizes),"coefs":[c.tolist() for c in model.coefs_],"biases":[b.tolist() for b in model.intercepts_],"iterations":int(model.n_iter_),"train_metrics":{"mse":float(mean_squared_error(y,pred)),"rmse":float(mean_squared_error(y,pred)**.5),"mae":float(mean_absolute_error(y,pred)),"r2":float(r2_score(y,pred))}})
    plt.figure(); plt.scatter(x,y,label="true"); plt.plot(x,pred,"o-",label="prediction"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.title("Approximation y=x²"); savefig(PLOTS/"approx_true_vs_pred.png")
    plt.figure(); plt.plot(np.arange(1,len(model.loss_curve_)+1),model.loss_curve_); plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.title("Approximation training curve"); savefig(PLOTS/"approx_training_curve.png")
    plt.figure(); plt.axhline(0,color="black",lw=1); plt.stem(x,residual); plt.xlabel("x"); plt.ylabel("Residual"); plt.title("Approximation residuals"); savefig(PLOTS/"approx_residuals.png")

def classification():
    rng=np.random.default_rng(SEED); a=rng.normal(loc=(-1.2,-1.0),scale=.55,size=(20,2)); b=rng.normal(loc=(1.2,1.0),scale=.55,size=(20,2)); X=np.vstack([a,b]); labels=np.array(["A"]*20+["B"]*20)
    pd.DataFrame({"feature_1":X[:,0],"feature_2":X[:,1],"class":labels}).to_csv(DATA/"classification.csv",index=False)
    model=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(4,),activation="relu",solver="adam",max_iter=5000,random_state=SEED))
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED); scores=cross_validate(model,X,labels,cv=cv,scoring=["accuracy","precision_macro","recall_macro","f1_macro"])
    model.fit(X,labels); pred=model.predict(X); cm=confusion_matrix(labels,pred,labels=["A","B"])
    write_json(RESULTS/"classification_metrics.json", {"cv_accuracy_mean":float(scores["test_accuracy"].mean()),"cv_precision_macro_mean":float(scores["test_precision_macro"].mean()),"cv_recall_macro_mean":float(scores["test_recall_macro"].mean()),"cv_f1_macro_mean":float(scores["test_f1_macro"].mean()),"train_classification_report":classification_report(labels,pred,output_dict=True),"confusion_matrix":cm.tolist()})
    plt.figure(); plt.scatter(a[:,0],a[:,1],label="A"); plt.scatter(b[:,0],b[:,1],label="B"); plt.legend(); plt.title("Classification data"); savefig(PLOTS/"classification_data.png")
    plt.figure(); plt.scatter(X[:,0],X[:,1],c=(pred=="B"),cmap="coolwarm"); plt.title("Classification predictions"); savefig(PLOTS/"classification_predictions.png")
    plt.figure(); DecisionBoundaryDisplay.from_estimator(model,X,response_method="predict",alpha=.25); plt.scatter(X[:,0],X[:,1],c=(labels=="B"),cmap="coolwarm",edgecolor="k"); plt.title("Decision regions"); savefig(PLOTS/"classification_decision_regions.png")
    plt.figure(); plt.imshow(cm,cmap="Blues"); plt.xticks([0,1],["A","B"]); plt.yticks([0,1],["A","B"]); plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix"); [plt.text(j,i,str(cm[i,j]),ha="center",va="center") for i in range(2) for j in range(2)]; savefig(PLOTS/"classification_confusion_matrix.png")

def iris_demo():
    iris=load_iris(); names=[f"sepal_length, sepal_width, petal_length, petal_width".split(", ")[i] for i in range(4)]; df=pd.DataFrame(iris.data,columns=names); df["class"]=iris.target_names[iris.target]; df.to_csv(DATA/"iris.csv",index=False)
    model=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(8,),activation="relu",solver="adam",max_iter=5000,random_state=SEED)); cv=StratifiedKFold(5,shuffle=True,random_state=SEED); sc=cross_validate(model,iris.data,iris.target,cv=cv,scoring=["accuracy","precision_macro","recall_macro","f1_macro"]); model.fit(iris.data,iris.target); pred=model.predict(iris.data); cm=confusion_matrix(iris.target,pred)
    write_json(RESULTS/"iris_metrics.json", {"cv_accuracy_mean":float(sc["test_accuracy"].mean()),"cv_precision_macro_mean":float(sc["test_precision_macro"].mean()),"cv_recall_macro_mean":float(sc["test_recall_macro"].mean()),"cv_f1_macro_mean":float(sc["test_f1_macro"].mean()),"confusion_matrix":cm.tolist(),"classes":iris.target_names.tolist()})
    plt.figure(); plt.scatter(iris.data[:,2],iris.data[:,3],c=iris.target,cmap="viridis"); plt.xlabel("petal length"); plt.ylabel("petal width"); plt.title("Iris scatter"); savefig(PLOTS/"iris_scatter.png")
    plt.figure(); plt.imshow(cm,cmap="Blues"); plt.xticks(range(3),iris.target_names,rotation=25); plt.yticks(range(3),iris.target_names); plt.title("Iris confusion matrix"); [plt.text(j,i,str(cm[i,j]),ha="center",va="center") for i in range(3) for j in range(3)]; savefig(PLOTS/"iris_confusion_matrix.png")

if __name__ == "__main__":
    for p in (DATA,RESULTS,PLOTS): p.mkdir(exist_ok=True)
    xor(); approximation(); classification(); iris_demo(); print("Lab 04 experiments completed")

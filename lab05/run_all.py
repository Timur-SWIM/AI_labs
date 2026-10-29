#!/usr/bin/env python3
"""Run Lab 05 Hamming network experiments."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.hamming import bipolarize, hamming_distance, potentials
from src.maxnet_feedforward import maxnet_feedforward
from src.maxnet_recurrent import maxnet_recurrent

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'; RESULTS=ROOT/'results'; PLOTS=ROOT/'plots'; SEED=42; DPI=180
plt.rcParams.update({'figure.dpi':DPI,'savefig.dpi':DPI,'axes.grid':True})

def savefig(path): plt.tight_layout(); plt.savefig(path,dpi=DPI); plt.close()

def make_patterns():
    # Demonstration patterns because no teacher-provided variant files exist.
    patterns=np.array([
        [[0,1,0,1,0],[1,1,1,1,1],[0,1,0,1,0],[0,1,0,1,0],[0,1,0,1,0]],
        [[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]],
        [[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]],
    ],dtype=int)
    labels=['cross','square','diamond'];
    for arr,label in zip(patterns,labels): pd.DataFrame(arr).to_csv(DATA/f'{label}.csv',index=False,header=False)
    fig,axes=plt.subplots(1,3,figsize=(7,2.6))
    for ax,arr,label in zip(axes,patterns,labels): ax.imshow(arr,cmap='gray_r',vmin=0,vmax=1); ax.set_title(label); ax.set_xticks([]); ax.set_yticks([])
    savefig(PLOTS/'reference_patterns.png')
    return np.array([bipolarize(p) for p in patterns]),labels

def exact(patterns,labels):
    rows=[]
    for i,(p,label) in enumerate(zip(patterns,labels)):
        pot=potentials(p,patterns); fw,fwv,trace=maxnet_feedforward(pot); rw,rwv,steps,history=maxnet_recurrent(pot,.1)
        for j,v in enumerate(pot): rows.append({'input':label,'class':labels[j],'potential':v,'feedforward_winner':labels[fw],'recurrent_winner':labels[rw],'recurrent_steps':steps,'is_correct':j==i})
    pd.DataFrame(rows).to_csv(RESULTS/'exact_results.csv',index=False)
    return rows

def noisy(patterns,labels):
    rng=np.random.default_rng(SEED); rows=[]; trials=100; n=patterns.shape[1]
    for distance in range(n+1):
        correct_f=correct_r=0
        for source,label in enumerate(labels):
            for trial in range(trials):
                x=patterns[source].copy(); flips=rng.choice(n,size=distance,replace=False); x[flips]*=-1; pot=potentials(x,patterns); fw,_,_=maxnet_feedforward(pot); rw,_,steps,_=maxnet_recurrent(pot,.1); correct_f+=fw==source; correct_r+=rw==source
                rows.append({'distance':distance,'source':label,'trial':trial,'feedforward_prediction':labels[fw],'recurrent_prediction':labels[rw],'recurrent_steps':steps,'correct_feedforward':fw==source,'correct_recurrent':rw==source})
        rows.append({'distance':distance,'source':'MEAN','trial':trials*len(labels),'feedforward_accuracy':correct_f/(trials*len(labels)),'recurrent_accuracy':correct_r/(trials*len(labels))})
    all_df=pd.DataFrame(rows); all_df.to_csv(RESULTS/'noise_trials.csv',index=False); summary=all_df[all_df.source=='MEAN']; summary.to_csv(RESULTS/'noise_accuracy.csv',index=False)
    plt.figure(); plt.plot(summary.distance,summary.feedforward_accuracy,'o-',label='feed-forward'); plt.plot(summary.distance,summary.recurrent_accuracy,'s-',label='recurrent'); plt.xlabel('Hamming distance'); plt.ylabel('Accuracy'); plt.ylim(-.02,1.02); plt.legend(); plt.title('Noise robustness'); savefig(PLOTS/'accuracy_vs_hamming_distance.png')

def weight_noise(patterns,labels):
    rng=np.random.default_rng(SEED+1); rows=[]; trials=500
    for level in [0,.01,.05,.10,.20]:
        cf=cr=0
        for _ in range(trials):
            weights=patterns+rng.normal(0,level,size=patterns.shape)
            for source,p in enumerate(patterns):
                pot=np.array([float(np.dot(p,w)/(2*len(p))+.5) for w in weights]); fw,_,_=maxnet_feedforward(pot); rw,_,_,_=maxnet_recurrent(pot,.1); cf+=fw==source; cr+=rw==source
        rows.append({'perturbation':level,'feedforward_accuracy':cf/(trials*len(patterns)),'recurrent_accuracy':cr/(trials*len(patterns))})
    df=pd.DataFrame(rows); df.to_csv(RESULTS/'weight_perturbation.csv',index=False); plt.figure(); plt.plot(df.perturbation*100,df.feedforward_accuracy,'o-',label='feed-forward'); plt.plot(df.perturbation*100,df.recurrent_accuracy,'s-',label='recurrent'); plt.xlabel('Weight perturbation (%)'); plt.ylabel('Accuracy'); plt.ylim(-.02,1.02); plt.legend(); plt.title('Weight robustness'); savefig(PLOTS/'accuracy_vs_weight_perturbation.png')

def epsilon_sweep(patterns):
    rows=[]; input_pattern=patterns[0]; pot=potentials(input_pattern,patterns)
    for eps in [.01,.03,.05,.10,.20,.25,.40,.49]:
        winner,value,steps,_=maxnet_recurrent(pot,eps); rows.append({'epsilon':eps,'winner_index':winner,'winner_value':value,'iterations':steps,'correct':winner==0})
    df=pd.DataFrame(rows); df.to_csv(RESULTS/'epsilon_sweep.csv',index=False); plt.figure(); plt.plot(df.epsilon,df.iterations,'o-'); plt.xlabel('epsilon'); plt.ylabel('Iterations'); plt.title('MAXNET iterations vs epsilon'); savefig(PLOTS/'iterations_vs_epsilon.png')

def main():
    for p in (DATA,RESULTS,PLOTS): p.mkdir(exist_ok=True)
    patterns,labels=make_patterns(); exact(patterns,labels); noisy(patterns,labels); weight_noise(patterns,labels); epsilon_sweep(patterns)
    (RESULTS/'metadata.json').write_text(json.dumps({'random_seed':SEED,'classes':labels,'shape':[5,5],'variant_data_available':False},indent=2),encoding='utf-8'); print('Lab 05 experiments completed')
if __name__=='__main__': main()


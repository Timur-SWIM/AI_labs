"""Local-only Streamlit seminar for Lab 06."""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.training_engine import TrainingEngine, validate_layers, MAX_EPOCHS
from src.experiments import make_split, evaluate, record
from src.lessons import neuron_metrics, descent_step
from src.exports import csv_bytes, json_bytes, markdown_report, parity_png
from src.scenarios import SCENARIOS, success

st.set_page_config(page_title="MLP Playground — ЛР6", page_icon="🧠", layout="wide")

def notice(text): st.caption("Вычислено по текущему эксперименту · " + text)
def footer(observation, question):
    st.divider(); st.info("Что студент должен заметить: " + observation); st.markdown("**Проверочный вопрос:** " + question)
def init():
    if "split" not in st.session_state: st.session_state.split = make_split()
    if "engine" not in st.session_state: st.session_state.engine = TrainingEngine((2,7,3,2))
    if "experiments" not in st.session_state: st.session_state.experiments = []
    if "gd" not in st.session_state: st.session_state.gd = [-1.0]
    if "control" not in st.session_state: st.session_state.control = None
init()

st.title("🧠 MLP Playground — ЛР №6")
st.caption("Локальный учебный тренажёр: никаких аккаунтов, внешних API, телеметрии или сетевых запросов. Формулы меток: x=ρcosφ, y=ρsinφ. Проверьте соглашения в методичке.")
with st.expander("Маршрут занятия и режим преподавателя"):
    teacher = st.toggle("Режим преподавателя", key="teacher")
    st.table(pd.DataFrame([["0–5","Введение","зачем учить преобразование координат"],["5–12","Один нейрон","подобрать w,b"],["12–20","Градиентный спуск","сравнить lr"],["20–45","Живая MLP","обучить 2–7–3–2"],["50–80","Почини сеть","гипотеза и метрики"],["80–90","Мини-защита","вывод"]],columns=["Минуты","Этап","Действие"]))
    st.multiselect("Чек-лист студента", ["Один нейрон", "Градиентный спуск", "MLP", "Диагностика", "Экспорт"], key="check")
    if teacher: st.success("Подсказка: не используйте control для выбора параметров; обсуждайте validation и причину разрыва train/validation.")

tab1, tab2, tab3, tab4 = st.tabs(["1. Один нейрон", "2. Градиентный спуск", "3. Живая MLP", "4. Почини нейросеть"])
with tab1:
    st.subheader("ŷ = w·x + b; L = (y − ŷ)²")
    st.caption("Квадрат ошибки здесь без множителя 1/2.")
    c1,c2,c3=st.columns(3)
    w=c1.slider("Вес w", -5., 6., 1., .1, help="Наклон прямой.")
    b=c2.slider("Смещение b", -8., 8., 0., .1, help="Пересечение с осью y.")
    both=c3.toggle("Учитывать оба примера", help="Добавляет x=3, target=10.")
    m=neuron_metrics(w,b,both); c1.metric("ŷ", f"{m['prediction'][0]:.3f}"); c2.metric("Суммарная ошибка", f"{m['sse']:.4f}"); c3.metric("Средняя ошибка", f"{m['mse']:.4f}")
    xs=np.linspace(0,4,80); fig=go.Figure([go.Scatter(x=xs,y=w*xs+b,name="модель ŷ"),go.Scatter(x=m['x'],y=m['target'],mode="markers",marker_size=12,name="истинные точки")]); fig.update_layout(height=430,xaxis_title="x",yaxis_title="y"); st.plotly_chart(fig,use_container_width=True); notice("линия и ошибка")
    st.write("С двумя условиями единственное решение: **w=3, b=1**. С одним условием пар бесконечно много.")
    if st.button("Сбросить пример нейрона"): st.rerun()
    footer("Одна точка не определяет одновременно наклон и смещение; вторая независимая точка определяет их.","Что модель подбирает и почему одного обучающего примера недостаточно для однозначного нахождения двух параметров?")

with tab2:
    st.subheader("Градиентный спуск для L(w)=(w−2)²")
    lr=st.slider("Learning rate",.01,1.5,.2,.01,help="Длина шага вдоль антиградиента.")
    a,b,c=st.columns(3)
    if a.button("Один шаг"):
        st.session_state.gd.append(descent_step(st.session_state.gd[-1],lr)["next_w"])
    if b.button("10 шагов"):
        for _ in range(10): st.session_state.gd.append(descent_step(st.session_state.gd[-1],lr)["next_w"])
    if c.button("Сброс",key="reset_gd"): st.session_state.gd=[-1.]; st.rerun()
    rows=[{k:v for k,v in descent_step(v,lr).items() if k != "next_w"}|{"step":i} for i,v in enumerate(st.session_state.gd)]
    current=rows[-1]; st.metric("Текущий w",f"{current['w']:.6g}",f"градиент {current['gradient']:.4g}")
    curve=np.linspace(min(-4,min(st.session_state.gd)-1),max(8,max(st.session_state.gd)+1),250); fig=go.Figure([go.Scatter(x=curve,y=(curve-2)**2,name="L(w)"),go.Scatter(x=st.session_state.gd,y=[(q-2)**2 for q in st.session_state.gd],mode="lines+markers+text",text=list(range(len(rows))),name="шаги")]); fig.update_layout(height=380); st.plotly_chart(fig,use_container_width=True); notice("траектория оптимизации")
    st.line_chart(pd.DataFrame(rows).set_index("step")[["loss"]]); st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    if not np.isfinite(current["loss"]) or abs(current["w"])>1e12: st.error("Расходимость: получены слишком большие или нечисловые значения.")
    st.caption("Малый lr медленный; 0<lr<1 устойчив для этой квадратичной функции; lr=1 колеблется между −1 и 5; lr>1 расходится. При lr=0.5 первый шаг ведёт ровно в w=2.")
    footer("Градиент зависит от текущего положения, а learning rate меняет устойчивость и скорость.","Почему backpropagation — это не оптимизатор, а алгоритм вычисления градиентов?")

def engine_controls(prefix="mlp", defaults=None):
    defaults={"layers":(2,7,3,2),"optimizer":"adam","learning_rate":.01,"epochs":200,"init_scale":1.,"seed":42,"tolerance":.05, **(defaults or {})}
    preset=st.selectbox("Архитектура",["2-4-4-2","2-7-3-2","2-10-6-2","ручная"],index=["2-4-4-2","2-7-3-2","2-10-6-2"].index("-".join(map(str,defaults['layers']))) if "-".join(map(str,defaults['layers'])) in ["2-4-4-2","2-7-3-2","2-10-6-2"] else 3,key=prefix+"arch")
    manual=st.text_input("Скрытые слои через запятую", ",".join(map(str,defaults["layers"][1:-1])),key=prefix+"manual")
    try: layers=tuple(map(int, preset.split("-"))) if preset!="ручная" else validate_layers((2,*(int(x.strip()) for x in manual.split(",") if x.strip()),2))
    except ValueError as e: st.error(str(e)); layers=defaults["layers"]
    col=st.columns(4); opt=col[0].selectbox("Оптимизатор",["adam","sgd"],index=0 if defaults["optimizer"]=="adam" else 1,key=prefix+"opt"); lr=col[1].number_input("Learning rate",1e-5,2.,float(defaults["learning_rate"]),format="%.5f",key=prefix+"lr"); epochs=col[2].number_input("Эпох за порцию",1,MAX_EPOCHS,int(defaults["epochs"]),key=prefix+"epochs"); seed=int(col[3].number_input("Seed",0,999999,int(defaults["seed"]),key=prefix+"seed"))
    col=st.columns(2); init_scale=col[0].number_input("init_scale",.01,20.,float(defaults["init_scale"]),key=prefix+"scale"); tolerance=col[1].number_input("GF tolerance",.001,1.,float(defaults["tolerance"]),key=prefix+"tol")
    return dict(layers=layers,optimizer=opt,learning_rate=lr,epochs=int(epochs),seed=seed,init_scale=init_scale,tolerance=tolerance)

with tab3:
    st.subheader("Живая MLP: (ρ, φ) → (x, y)")
    st.write("Teacher labels вычисляются по формулам; сеть видит только пары обучающих примеров. Нормализация вычислена исключительно по train.")
    cfg=engine_controls(); stop80=st.toggle("Остановить при 80% GF",key="stop80",help="Остановка проверяет только train GF."); top=st.columns(5)
    if top[0].button("Перезапустить с тем же seed / Reset"):
        st.session_state.engine=TrainingEngine(**{k:cfg[k] for k in cfg if k!="epochs"}); st.session_state.control=None; st.rerun()
    if top[1].button("Обучить / Продолжить"):
        e=st.session_state.engine
        same=(e.layers==cfg["layers"] and e.optimizer==cfg["optimizer"] and e.learning_rate==cfg["learning_rate"] and e.seed==cfg["seed"] and e.init_scale==cfg["init_scale"])
        if not same: st.warning("Настройки изменены: нажмите Reset для нового честного эксперимента.")
        else:
            e.stopped=False
            x,y=st.session_state.split.arrays(st.session_state.split.train); xv,yv=st.session_state.split.arrays(st.session_state.split.validation); e.train_epochs(cfg["epochs"],x,y,xv,yv, .8 if stop80 else None)
    if top[2].button("Остановить"): st.session_state.engine.stopped=True
    if top[3].button("Сохранить эксперимент"):
        st.session_state.experiments.append(record(st.session_state.engine,evaluate(st.session_state.engine,st.session_state.split))); st.session_state.experiments=st.session_state.experiments[-5:]
    if top[4].button("Итоговая оценка control"):
        st.session_state.control=evaluate(st.session_state.engine,st.session_state.split,True)["control"]
    e=st.session_state.engine; met=evaluate(e,st.session_state.split); k=st.columns(6); k[0].metric("Эпоха",e.epoch); k[1].metric("Параметры",e.parameter_count); k[2].metric("MSE train",f"{met['train']['mse']:.5f}"); k[3].metric("MSE val",f"{met['validation']['mse']:.5f}"); k[4].metric("GF/BF val",f"{met['validation']['gf']}/{met['validation']['bf']}"); k[5].metric("Control MSE", "—" if not st.session_state.control else f"{st.session_state.control['mse']:.5f}")
    if st.session_state.control: st.warning("Control — итоговая оценка; не выбирайте по ней гиперпараметры.")
    xall,yall=st.session_state.split.arrays(st.session_state.split.validation); pred=e.predict(xall); i=st.slider("Тестовый пример для вектора",0,len(yall)-1,0); dx,dy=np.abs(yall[i]-pred[i]); st.caption(f"Текущие ошибки: |dx|={dx:.5f}, |dy|={dy:.5f}; статус: {'остановлен' if e.stopped else 'готов'}")
    left,right=st.columns(2)
    with left:
        h=pd.DataFrame(e.history)
        if not h.empty: st.plotly_chart(go.Figure([go.Scatter(x=h.epoch,y=h.train_loss,name="train"),go.Scatter(x=h.epoch,y=h.validation_loss,name="validation")]).update_layout(height=360,title="Loss vs Epoch"),use_container_width=True); notice("loss текущих весов")
        else: st.info("Нажмите «Обучить»: история появится после реальных обновлений весов.")
        vf=go.Figure(); vf.add_scatter(x=[0,yall[i,0]],y=[0,yall[i,1]],mode="lines+markers",name="истинный"); vf.add_scatter(x=[0,pred[i,0]],y=[0,pred[i,1]],mode="lines+markers",name="предсказанный"); vf.add_scatter(x=[yall[i,0],pred[i,0]],y=[yall[i,1],pred[i,1]],mode="lines",name="ошибка"); vf.update_layout(height=360,title="Геометрия вектора",xaxis_scaleanchor="y"); st.plotly_chart(vf,use_container_width=True)
    with right:
        parity=go.Figure([go.Scatter(x=yall[:,0],y=pred[:,0],mode="markers",name="x"),go.Scatter(x=yall[:,1],y=pred[:,1],mode="markers",name="y"),go.Scatter(x=[0,1],y=[0,1],name="ŷ=y",line=dict(dash="dash"))]); parity.update_layout(height=360,title="True vs Predicted",xaxis_title="true",yaxis_title="predicted"); st.plotly_chart(parity,use_container_width=True)
        for layer,wmat in enumerate(e.weights):
            st.plotly_chart(go.Figure(go.Heatmap(z=wmat,colorscale="RdBu",zmid=0,zmin=-2,zmax=2)).update_layout(height=220,title=f"Матрица весов W{layer+1} (фиксированная шкала ±2)"),use_container_width=True)
    st.subheader("Сравнение и экспорт")
    if st.session_state.experiments:
        comparison=pd.DataFrame([{k:v for k,v in r.items() if k!="history"} for r in st.session_state.experiments]); st.dataframe(comparison,use_container_width=True); st.download_button("CSV экспериментов",csv_bytes(comparison),"experiments.csv","text/csv"); st.download_button("JSON экспериментов",json_bytes(st.session_state.experiments),"experiments.json","application/json"); st.download_button("Краткий Markdown",markdown_report(st.session_state.experiments[-1]),"report.md","text/markdown"); st.download_button("PNG текущего parity plot",parity_png(yall,pred),"parity.png","image/png")
    with st.expander("Экспорт в Orange"):
        f=st.session_state.split.frame; st.download_button("coord.csv",csv_bytes(f),"coord.csv","text/csv"); st.download_button("coord_x.csv",csv_bytes(f[["rho","phi","x"]]),"coord_x.csv","text/csv"); st.download_button("coord_y.csv",csv_bytes(f[["rho","phi","y"]]),"coord_y.csv","text/csv"); st.write("Orange: File → Select Columns → Neural Network → Predictions / Test and Score. Обычно нужны две отдельные одноцелевые модели для x и y, поэтому метрики не обязаны совпадать с единой двухвыходной NumPy MLP.")
    footer("Loss, векторы и heatmap меняются от текущих весов; низкий train loss не подтверждает качество на новых точках.","Почему низкий train loss не гарантирует хорошее качество на новых точках?")

with tab4:
    st.subheader("Почини нейросеть")
    name=st.selectbox("Сценарий",list(SCENARIOS)); scenario=SCENARIOS[name]
    if st.button("Загрузить проблему"):
        st.session_state.scenario_cfg=copy.deepcopy(scenario); st.session_state.scenario_before=None
    cfg=engine_controls("scenario",getattr(st.session_state,"scenario_cfg",scenario)); st.write("Цель:",scenario["goal"])
    if st.button("Запустить сценарий"):
        eng=TrainingEngine(**{k:cfg[k] for k in cfg if k!="epochs"}); split=make_split(cfg["seed"],train_size=30 if name.startswith("C") else 60,val_size=20); x,y=split.arrays(split.train); xv,yv=split.arrays(split.validation); eng.train_epochs(cfg["epochs"],x,y,xv,yv); after=evaluate(eng,split); st.session_state.scenario_after=after
        if st.session_state.scenario_before is None: st.session_state.scenario_before=after
    after=getattr(st.session_state,"scenario_after",None); before=getattr(st.session_state,"scenario_before",None)
    if after:
        ok,message=success(name,before,after); st.success("Критерий выполнен." if ok else "Пока не выполнен. "+message); st.json(after)
    st.text_area("Гипотеза / Что изменил / Что увидел / Вывод",key="hypothesis")
    if st.session_state.get("teacher"): st.info("Объяснение: A — поднимите lr постепенно; B — уменьшите lr; C — проверьте размер модели/данных и сравните validation, а не рисуйте переобучение.")
    footer("Симптомы и успех считаются из реального запуска с фиксированными настройками, не из готовой картинки.","Какая метрика показывает качество на новых данных и почему одного train loss недостаточно?")

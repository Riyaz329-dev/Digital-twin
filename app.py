# -*- coding: utf-8 -*-
# -------------------------------------------------------
# app.py — Digital Twin Framework for Real-Time Diabetes Management
# -------------------------------------------------------
# Expected files:
#   best_model.pkl, feature_cols.pkl, decision_threshold.pkl, Collected_cleaned.csv
#
# Run:
#   pip install streamlit pandas plotly scikit-learn xgboost joblib
#   streamlit run app.py
# -------------------------------------------------------

import os, time, joblib, numpy as np, pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ---------- GLOBAL CONFIG ----------
st.set_page_config(
    page_title="Digital Twin: Real-Time Diabetes Management",
    page_icon="🩺",
    layout="wide",
)

# ---------- HELPERS: LOADERS & CACHE ----------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load("best_model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    threshold = joblib.load("decision_threshold.pkl")
    return model, feature_cols, float(threshold)

@st.cache_data
def load_collected() -> pd.DataFrame:
    df = pd.read_csv("Collected_cleaned.csv")
    if "timestamp" not in df.columns:
        raise ValueError("Collected_cleaned.csv must contain a 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "patient_id" not in df.columns:
        raise ValueError("Collected_cleaned.csv must contain 'patient_id'.")
    start_date_hist = df["timestamp"].dt.normalize().min()
    today_norm = pd.Timestamp.now().normalize()
    delta_to_today = today_norm - start_date_hist
    df["shifted_ts"] = df["timestamp"] + delta_to_today
    return df.sort_values(["patient_id","shifted_ts"]).reset_index(drop=True)

def get_patient_list(df: pd.DataFrame):
    return sorted(df["patient_id"].astype(str).unique().tolist())

def advise(glucose: float, hba1c: float) -> str:
    if glucose < 140 and hba1c < 6:
        return "🟢 Normal: Maintain healthy diet and regular exercise."
    elif 140 <= glucose < 180 or (6 <= hba1c < 7):
        return "⚠️ Slightly elevated: Recheck in a few hours and limit sugar intake."
    elif glucose >= 180 or hba1c >= 7:
        return "🔴 High: Monitor closely and consult your clinician if persistent."
    return "ℹ️ Unable to determine — please verify readings."

def predict_row(model, feature_cols, row: pd.Series) -> float:
    x = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
    return float(model.predict_proba(x)[0][1])

def compute_kpis(g: pd.DataFrame, threshold: float):
    tir = (g["blood_glucose_level"] < 180).mean() * 100 if not g.empty else None
    alerts = int((g.get("proba", pd.Series([])) >= threshold).sum()) if "proba" in g else 0
    mean_glucose = float(g["blood_glucose_level"].mean()) if not g.empty else np.nan
    max_prob = float(g.get("proba", pd.Series([0.0])).max()) if "proba" in g else np.nan
    return tir, alerts, mean_glucose, max_prob

def ensure_session():
    keys = ["autoplay","last_tick","stream_index","log_df"]
    defaults = [False,0.0,{},pd.DataFrame(columns=[
        "patient_id","historical_ts","shifted_ts","arrival_ts",
        "shown_ts","blood_glucose_level","HbA1c_level",
        "proba","pred","status","advice"
    ])]
    for k,v in zip(keys,defaults):
        if k not in st.session_state: st.session_state[k]=v

def init_patient_pointer(pid: str, df: pd.DataFrame):
    if pid not in st.session_state.stream_index:
        st.session_state.stream_index[pid]=0

def append_to_csv(path: str, df_new: pd.DataFrame):
    header = not os.path.exists(path)
    df_new.to_csv(path, mode="a", header=header, index=False)

def advance_one_tick(pid: str, df: pd.DataFrame, model, feature_cols, threshold: float):
    g = df[df["patient_id"].astype(str)==str(pid)].sort_values("shifted_ts")
    if g.empty: return False
    i = st.session_state.stream_index.get(pid,0)
    if i>=len(g): return False
    row = g.iloc[i]
    proba = predict_row(model, feature_cols, row)
    pred = int(proba>=threshold)
    status = "DIABETIC" if pred==1 else "NORMAL"
    rec = {
        "patient_id": row["patient_id"],
        "historical_ts": row["timestamp"],
        "shifted_ts": row["shifted_ts"],
        "arrival_ts": pd.Timestamp.now(),
        "shown_ts": row["shifted_ts"],
        "blood_glucose_level": float(row["blood_glucose_level"]),
        "HbA1c_level": float(row["HbA1c_level"]),
        "proba": proba, "pred": pred, "status": status,
        "advice": advise(float(row["blood_glucose_level"]), float(row["HbA1c_level"]))
    }
    st.session_state.log_df = pd.concat([st.session_state.log_df,pd.DataFrame([rec])],ignore_index=True)
    st.session_state.stream_index[pid]=i+1
    try: append_to_csv("live_predictions_log.csv", pd.DataFrame([rec]))
    except: pass
    return True

# ---------- CLOCK PANEL ----------
def clock_panel(selected_pid: str|None, df: pd.DataFrame, title="⏱️ Live vs Simulated Time"):
    st.markdown(f"### {title}")
    c1,c2=st.columns(2)
    c1.metric("System Clock (local)", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    sim_label="—"
    if selected_pid is not None:
        log=st.session_state.log_df
        g_log=log[log["patient_id"].astype(str)==str(selected_pid)].sort_values("shown_ts")
        if not g_log.empty:
            sim_label=pd.to_datetime(g_log.iloc[-1]["shown_ts"]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            g_src=df[df["patient_id"].astype(str)==str(selected_pid)].sort_values("shifted_ts")
            i=st.session_state.stream_index.get(selected_pid,0)
            if not g_src.empty:
                if i<len(g_src): sim_label=pd.to_datetime(g_src.iloc[i]["shifted_ts"]).strftime("%Y-%m-%d %H:%M:%S")
                else: sim_label=pd.to_datetime(g_src.iloc[-1]["shifted_ts"]).strftime("%Y-%m-%d %H:%M:%S")
    c2.metric(f"Simulated Time (dataset) — Patient {selected_pid if selected_pid else ''}", sim_label)
    st.caption("System Clock = real local time • Simulated Time = dataset timestamp being replayed.")
# ---------- FORECAST HELPERS ----------
def build_patient_series(df: pd.DataFrame, pid: str) -> pd.DataFrame:
    g = df[df["patient_id"].astype(str)==str(pid)].sort_values("shifted_ts").copy()
    return g[["shifted_ts","blood_glucose_level","HbA1c_level"]].rename(
        columns={"shifted_ts":"ts","blood_glucose_level":"glucose","HbA1c_level":"hba1c"}
    )

def ewma_forecast(series: pd.Series, steps: int, alpha=0.5, noise_std=3.0) -> np.ndarray:
    if len(series)==0: return np.array([])
    level=series.iloc[-1]; ew=series.ewm(alpha=alpha).mean().iloc[-1]
    preds=[]; current=level
    for _ in range(steps):
        current=0.6*current+0.4*ew+np.random.normal(0,noise_std)
        preds.append(max(current,0))
    return np.array(preds)

def apply_scenario(glucose_arr: np.ndarray, scenario: str) -> np.ndarray:
    factors={"Maintain (no change)":1.00,"Improve (diet/med −10%)":0.90,
             "Strong improve (−20%)":0.80,"Worsen (+10%)":1.10}
    f=factors.get(scenario,1.00)
    return np.clip(glucose_arr*f,0,None)

# ---------- DAY-BOUNDED FORECAST ----------
def simulate_future_day(pid: str, df_src: pd.DataFrame, model, feature_cols: list, threshold: float,
                        session_date: pd.Timestamp, horizon_hours=24, freq_minutes=60,
                        method="EWMA", scenario="Maintain (no change)"):
    g=df_src[df_src["patient_id"].astype(str)==str(pid)].copy()
    g["shifted_ts"]=pd.to_datetime(g["shifted_ts"])
    day=pd.to_datetime(session_date).date()
    day_hist=g[g["shifted_ts"].dt.date==day].sort_values("shifted_ts")
    if day_hist.empty: return pd.DataFrame(),0.0

    last_time=day_hist["shifted_ts"].max()
    day_start=pd.Timestamp.combine(day,pd.Timestamp.min.time())
    day_end=day_start+pd.Timedelta(days=1)

    freq_td=pd.to_timedelta(f"{int(freq_minutes)}min")
    req_steps=max(1,int((horizon_hours*60)//freq_minutes))
    remaining=(day_end-last_time)
    max_steps_today=max(int(remaining//freq_td),0)
    steps=min(req_steps,max_steps_today)
    if steps<=0: return pd.DataFrame(),0.0

    future_index=pd.date_range(start=last_time+freq_td,periods=steps,freq=freq_td)

    if method=="Hold-Last":
        base=np.full(steps,day_hist["blood_glucose_level"].iloc[-1])
        noise=np.random.normal(0,2.0,size=steps)
        g_fore=np.clip(base+noise,0,None)
    else:
        g_fore=ewma_forecast(day_hist["blood_glucose_level"],steps=steps,alpha=0.5,noise_std=3.0)
    g_fore=apply_scenario(g_fore,scenario)
    hba1c_last=float(day_hist["HbA1c_level"].iloc[-1])

    last_row=g.sort_values("shifted_ts").iloc[-1].copy()
    fut=pd.DataFrame({
        "patient_id":pid,
        "timestamp":future_index,
        "shifted_ts":future_index,
        "HbA1c_level":np.full(steps,hba1c_last),
        "blood_glucose_level":g_fore
    })
    for col in feature_cols:
        if col not in ["HbA1c_level","blood_glucose_level"]:
            fut[col]=last_row.get(col,0)

    proba=model.predict_proba(fut[feature_cols])[:,1]
    fut["proba"]=proba
    fut["pred"]=(proba>=threshold).astype(int)
    fut["status"]=np.where(fut["pred"]==1,"DIABETIC","NORMAL")
    fut["advice"]=[advise(g,h) for g,h in zip(fut["blood_glucose_level"],fut["HbA1c_level"])]
    fut["is_forecast"]=True
    used_hours=steps*(freq_td/pd.Timedelta(hours=1))
    return fut,float(used_hours)

# ---------- VISUALS ----------
def plot_patient_trend(df_show: pd.DataFrame, threshold: float, title: str):
    if df_show.empty:
        st.info("No data yet to plot."); return
    colors=df_show["status"].map(lambda s:"#d62728" if s=="DIABETIC" else "#2ca02c")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df_show["shown_ts"],y=df_show["blood_glucose_level"],
                             mode="lines+markers",
                             marker=dict(size=8,color=colors,line=dict(width=1,color="black")),
                             line=dict(width=2),name="Glucose (mg/dL)"))
    fig.add_hrect(y0=70,y1=180,fillcolor="rgba(0,200,0,0.05)",line_width=0)
    fig.add_hline(y=180,line_dash="dash",line_color="orange",annotation_text="180 mg/dL")
    fig.update_layout(title=title,xaxis_title="Time",yaxis_title="Blood Glucose (mg/dL)",height=420)
    st.plotly_chart(fig,use_container_width=True)

def cohort_tiles(df_log: pd.DataFrame, pids: list, threshold: float):
    cols=st.columns(5)
    for idx,pid in enumerate(pids):
        g=df_log[df_log["patient_id"].astype(str)==str(pid)]
        if g.empty:
            status,prob,last_g,dot,t="—","—","—","⚪","—"
        else:
            last=g.sort_values("shown_ts").iloc[-1]
            status,last_g=int(last["pred"]),int(last["blood_glucose_level"])
            status_txt="DIABETIC" if status==1 else "NORMAL"
            prob=f"{last['proba']:.2f}"
            dot="🟢" if status_txt=="NORMAL" else "🔴"
            t=pd.to_datetime(last["shown_ts"]).strftime("%d %b %H:%M")
        with cols[idx%5]:
            st.metric(label=f"{dot} Patient {pid}",value=f"Status: {status_txt}",delta=f"Prob: {prob}")
            st.caption(f"Last: {last_g} mg/dL • {t}")

# ---------- PAGES ----------
def page_all_patients(df, model, feature_cols, threshold):
    st.markdown("## 🏥 All Patients — Command Center")
    pids=get_patient_list(df)
    with st.sidebar:
        st.markdown("### ▶️ Stream Controls")
        autoplay=st.toggle("Auto-play stream",value=st.session_state.autoplay)
        st.session_state.autoplay=autoplay
        refresh=st.slider("Auto interval (sec)",0.5,3.0,1.0,0.5)
        target_pid=st.selectbox("Select patient",pids)
        init_patient_pointer(target_pid,df)
        col_a,col_b=st.columns(2)
        if col_a.button("Advance one tick"):
            advance_one_tick(target_pid,df,model,feature_cols,threshold); st.rerun()
        if col_b.button("Reset stream"):
            st.session_state.stream_index[target_pid]=0
            st.session_state.log_df=st.session_state.log_df[st.session_state.log_df["patient_id"]!=target_pid]
            st.rerun()
    clock_panel(target_pid,df)
    now=time.time()
    if st.session_state.autoplay and now-st.session_state.last_tick>=refresh:
        if advance_one_tick(target_pid,df,model,feature_cols,threshold):
            st.session_state.last_tick=now; st.rerun()
    st.markdown("#### Cohort Overview")
    log=st.session_state.log_df.copy(); cohort_tiles(log,pids,threshold)
    st.markdown("#### Recent Alerts")
    if log.empty: st.info("No alerts yet."); return
    alerts=log[log["proba"]>=threshold].sort_values("shown_ts",ascending=False).head(50)
    st.dataframe(alerts[["patient_id","shown_ts","blood_glucose_level","HbA1c_level","proba","status","advice"]]
                 .rename(columns={"blood_glucose_level":"glucose","HbA1c_level":"hba1c"}),
                 use_container_width=True,hide_index=True)

def page_patient_twin(df, model, feature_cols, threshold):
    st.markdown("## 👤 Patient Twin — Real-Time Cockpit")
    pids=get_patient_list(df)
    with st.sidebar:
        pid=st.selectbox("Patient",pids)
        init_patient_pointer(pid,df)
        autoplay=st.toggle("Auto-play",value=st.session_state.autoplay)
        st.session_state.autoplay=autoplay
        refresh=st.slider("Interval (sec)",0.5,3.0,1.0,0.5)
        col_a,col_b=st.columns(2)
        if col_a.button("Next tick"): advance_one_tick(pid,df,model,feature_cols,threshold); st.rerun()
        if col_b.button("Reset"): st.session_state.stream_index[pid]=0
        st.markdown(f"**Threshold:** {threshold:.3f}")

    clock_panel(pid,df)
    now=time.time()
    if st.session_state.autoplay and now-st.session_state.last_tick>=refresh:
        if advance_one_tick(pid,df,model,feature_cols,threshold):
            st.session_state.last_tick=now; st.rerun()

    log=st.session_state.log_df.copy()
    g=log[log["patient_id"].astype(str)==str(pid)].sort_values("shown_ts")

    c1,c2,c3,c4=st.columns(4)
    if g.empty:
        [c.metric(x,"—") for c,x in zip([c1,c2,c3,c4],["Glucose","HbA1c","Prob","Status"])]
    else:
        last=g.iloc[-1]
        c1.metric("Latest Glucose",f"{int(last['blood_glucose_level'])} mg/dL")
        c2.metric("Latest HbA1c",f"{float(last['HbA1c_level']):.2f}")
        c3.metric("Risk Prob",f"{float(last['proba']):.3f}")
        c4.metric("Status",last["status"],help=last["advice"])

    plot_patient_trend(g,threshold,f"Patient {pid} — Glucose trend")

    k1,k2,k3,k4=st.columns(4)
    tir,alerts,mean_glu,max_p=compute_kpis(g,threshold)
    k1.metric("Time-in-Range %",f"{0 if tir is None else tir:.1f}%")
    k2.metric("Alerts",f"{alerts}")
    k3.metric("Mean Glucose",f"{mean_glu:.0f} mg/dL" if not np.isnan(mean_glu) else "—")
    k4.metric("Max Prob",f"{max_p:.2f}" if not np.isnan(max_p) else "—")

    st.markdown("#### Advice Feed")
    if g.empty: st.info("No events yet.")
    else:
        st.dataframe(g[["shown_ts","blood_glucose_level","HbA1c_level","status","advice","proba"]]
                     .rename(columns={"shown_ts":"time","blood_glucose_level":"glucose","HbA1c_level":"hba1c"})
                     .tail(10),use_container_width=True,hide_index=True)

    # ---------- 🔮 FUTURE SIMULATOR (DAY-BOUNDED) ----------
    st.markdown("---"); st.markdown("### 🔮 Future Simulator (Forecast)")
    df_pid=df[df["patient_id"].astype(str)==str(pid)].copy()
    df_pid["shifted_ts"]=pd.to_datetime(df_pid["shifted_ts"])
    days=sorted(df_pid["shifted_ts"].dt.date.unique())
    if not days: st.info("No data for this patient."); return
    session_date=st.date_input("Session date",value=days[-1],min_value=days[0],max_value=days[-1])
    col1,col2,col3,col4=st.columns(4)
    horizon=col1.slider("Horizon (hours)",1,72,24,1)
    freq=col2.selectbox("Frequency",[5,10,15,30,60],index=2,format_func=lambda m:f"{m} min")
    method=col3.selectbox("Method",["EWMA","Hold-Last"])
    scenario=col4.selectbox("Scenario",["Maintain (no change)","Improve (diet/med −10%)",
                                        "Strong improve (−20%)","Worsen (+10%)"])
    if st.button("Generate forecast"):
        fut,used_hours=simulate_future_day(pid,df,model,feature_cols,threshold,
                                           session_date=session_date,horizon_hours=horizon,
                                           freq_minutes=freq,method=method,scenario=scenario)
        mask=df_pid["shifted_ts"].dt.date==pd.to_datetime(session_date).date()
        hist_day=df_pid.loc[mask,["shifted_ts","blood_glucose_level"]].sort_values("shifted_ts")
        if fut.empty:
            st.info("No time left in the selected day at this frequency."); return
        if used_hours<float(horizon):
            st.warning(f"Requested {horizon}h, but only {used_hours:.1f}h fit before midnight (clipped).")
        st.dataframe(fut[["shifted_ts","blood_glucose_level","HbA1c_level","proba","status","advice"]]
                     .rename(columns={"shifted_ts":"time","blood_glucose_level":"glucose","HbA1c_level":"hba1c"})
                     .head(20),use_container_width=True,hide_index=True)
        fig=go.Figure()
        if not hist_day.empty:
            fig.add_trace(go.Scatter(x=hist_day["shifted_ts"],y=hist_day["blood_glucose_level"],
                                     mode="lines+markers",name="Historical",
                                     line=dict(width=2),marker=dict(size=7,line=dict(width=1,color="black"))))
        fig.add_trace(go.Scatter(x=fut["shifted_ts"],y=fut["blood_glucose_level"],
                                 mode="lines+markers",name="Forecast",
                                 line=dict(width=2,dash="dash"),marker=dict(size=7)))
        fig.add_hrect(y0=70,y1=180,fillcolor="rgba(0,200,0,0.05)",line_width=0)
        fig.add_hline(y=180,line_dash="dash",line_color="orange",annotation_text="180 mg/dL")
        fig.update_layout(title=f"Patient {pid} — {pd.to_datetime(session_date).date()} (Day-bounded forecast)",
                          xaxis_title="Time",yaxis_title="Blood Glucose (mg/dL)",height=420)
        st.plotly_chart(fig,use_container_width=True)
        alerts_fut=fut[fut["proba"]>=threshold][["shifted_ts","blood_glucose_level","HbA1c_level","proba","status","advice"]]
        if alerts_fut.empty: st.success("No forecasted alerts in this horizon.")
        else:
            st.warning(f"{len(alerts_fut)} forecasted alert(s).")
            st.dataframe(alerts_fut.rename(columns={"shifted_ts":"time","blood_glucose_level":"glucose",
                                                    "HbA1c_level":"hba1c"}),
                         use_container_width=True,hide_index=True)
    st.caption("⚠️ Educational prototype — not for medical use.")

def page_model_threshold(df, model, feature_cols, threshold):
    st.markdown("## 🧠 Model & Threshold — Twin Brain")
    try:
        imp=getattr(model,"feature_importances_",None)
        if imp is not None:
            fi=pd.DataFrame({"feature":feature_cols,"importance":imp}).sort_values("importance",ascending=False).head(15)
            st.plotly_chart(px.bar(fi,x="importance",y="feature",orientation="h",title="Top Features"),use_container_width=True)
        else: st.info("Model does not expose feature_importances_.")
    except Exception as e: st.warning(f"Feature importances unavailable: {e}")
    st.markdown("### Threshold Tuning")
    new_thr=st.slider("Threshold",0.05,0.95,float(threshold),0.01)
    try:
        df_eval=df.dropna(subset=feature_cols+["diabetes"]).copy()
        X,y=df_eval[feature_cols],df_eval["diabetes"].astype(int)
        proba=model.predict_proba(X)[:,1]; y_pred=(proba>=new_thr).astype(int)
        acc,prec,rec,f1,auc=accuracy_score(y,y_pred),precision_score(y,y_pred,zero_division=0),\
                             recall_score(y,y_pred,zero_division=0),f1_score(y,y_pred,zero_division=0),\
                             roc_auc_score(y,proba)
        for val,label in zip([acc,prec,rec,f1,auc],["Accuracy","Precision","Recall","F1","ROC-AUC"]):
            st.metric(label,f"{val:.3f}")
        with st.expander("Confusion Matrix"):
            st.write(pd.DataFrame(confusion_matrix(y,y_pred),
                    index=["Actual 0","Actual 1"],columns=["Pred 0","Pred 1"]))
    except Exception as e: st.warning(f"Eval error: {e}")
    if st.button("💾 Save this threshold"):
        joblib.dump(float(new_thr),"decision_threshold.pkl")
        st.success(f"Saved threshold {new_thr:.3f}. Reload app to apply.")

def page_data_logs():
    st.markdown("## 📜 Data & Logs — Twin Memory")
    if st.session_state.log_df.empty: st.info("No in-session log."); 
    else: st.dataframe(st.session_state.log_df.tail(200),use_container_width=True,hide_index=True)
    if os.path.exists("live_predictions_log.csv"):
        disk

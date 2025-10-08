# -*- coding: utf-8 -*-
# app.py — Digital Twin Framework for Real-Time Diabetes Management
# -------------------------------------------------------
# Files expected in the same folder:
#  - best_model.pkl
#  - feature_cols.pkl
#  - decision_threshold.pkl
#  - Collected_cleaned.csv
#
# Run:
#   pip install streamlit pandas plotly scikit-learn xgboost joblib
#   streamlit run app.py

import os
import time
import joblib
import numpy as np
import pandas as pd
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
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError("Collected_cleaned.csv must contain a 'timestamp' column.")
    start_date_hist = df["timestamp"].dt.normalize().min()
    today_norm = pd.Timestamp.now().normalize()
    delta_to_today = today_norm - start_date_hist
    df["shifted_ts"] = df["timestamp"] + delta_to_today
    if "patient_id" not in df.columns:
        raise ValueError("Collected_cleaned.csv must contain 'patient_id'.")
    return df.sort_values(["patient_id", "shifted_ts"]).reset_index(drop=True)

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
    tir = None
    if not g.empty:
        tir = float((g["blood_glucose_level"] < 180).mean() * 100.0)
    alerts = int((g.get("proba", pd.Series([])) >= threshold).sum()) if "proba" in g else 0
    mean_glucose = float(g["blood_glucose_level"].mean()) if not g.empty else np.nan
    max_prob = float(g.get("proba", pd.Series([0.0])).max()) if "proba" in g else np.nan
    return tir, alerts, mean_glucose, max_prob

def ensure_session():
    if "autoplay" not in st.session_state:
        st.session_state.autoplay = False
    if "last_tick" not in st.session_state:
        st.session_state.last_tick = 0.0
    if "stream_index" not in st.session_state:
        st.session_state.stream_index = {}
    if "log_df" not in st.session_state:
        st.session_state.log_df = pd.DataFrame(columns=[
            "patient_id","historical_ts","shifted_ts","arrival_ts",
            "shown_ts","blood_glucose_level","HbA1c_level",
            "proba","pred","status","advice"
        ])

def init_patient_pointer(pid: str, df: pd.DataFrame):
    if pid not in st.session_state.stream_index:
        st.session_state.stream_index[pid] = 0

def append_to_csv(path: str, df_new: pd.DataFrame):
    header = not os.path.exists(path)
    df_new.to_csv(path, mode="a", header=header, index=False)

def advance_one_tick(pid: str, df: pd.DataFrame, model, feature_cols, threshold: float):
    g = df[df["patient_id"].astype(str) == str(pid)].sort_values("shifted_ts")
    if g.empty:
        return False
    i = st.session_state.stream_index.get(pid, 0)
    if i >= len(g):
        return False
    row = g.iloc[i]
    proba = predict_row(model, feature_cols, row)
    pred = 1 if proba >= threshold else 0
    status = "DIABETIC" if pred == 1 else "NORMAL"
    rec = {
        "patient_id": row["patient_id"],
        "historical_ts": row["timestamp"],
        "shifted_ts": row["shifted_ts"],
        "arrival_ts": pd.Timestamp.now(),
        "shown_ts": row["shifted_ts"],
        "blood_glucose_level": float(row["blood_glucose_level"]),
        "HbA1c_level": float(row["HbA1c_level"]),
        "proba": proba,
        "pred": pred,
        "status": status,
        "advice": advise(float(row["blood_glucose_level"]), float(row["HbA1c_level"]))
    }
    st.session_state.log_df = pd.concat([st.session_state.log_df, pd.DataFrame([rec])], ignore_index=True)
    st.session_state.stream_index[pid] = i + 1
    try:
        append_to_csv("live_predictions_log.csv", pd.DataFrame([rec]))
    except Exception:
        pass
    return True

# ---------- CLOCK PANEL ----------
def clock_panel(selected_pid: str | None, df: pd.DataFrame, title: str = "⏱️ Live vs Simulated Time"):
    """Show real system time vs simulated dataset time."""
    st.markdown(f"### {title}")
    c1, c2 = st.columns(2)
    system_now = pd.Timestamp.now()
    c1.metric("System Clock (local)", system_now.strftime("%Y-%m-%d %H:%M:%S"))
    sim_label = "—"
    if selected_pid is not None:
        log = st.session_state.log_df
        g_log = log[log["patient_id"].astype(str) == str(selected_pid)].sort_values("shown_ts")
        if not g_log.empty:
            sim_label = pd.to_datetime(g_log.iloc[-1]["shown_ts"]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            g_src = df[df["patient_id"].astype(str) == str(selected_pid)].sort_values("shifted_ts")
            i = st.session_state.stream_index.get(selected_pid, 0)
            if not g_src.empty:
                if i < len(g_src):
                    sim_label = pd.to_datetime(g_src.iloc[i]["shifted_ts"]).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    sim_label = pd.to_datetime(g_src.iloc[-1]["shifted_ts"]).strftime("%Y-%m-%d %H:%M:%S")
    c2.metric(f"Simulated Time (dataset) — Patient {selected_pid if selected_pid else ''}", sim_label)
    st.caption("System Clock = real local time • Simulated Time = dataset timestamp being replayed.")

# ---------- VISUALS ----------
def plot_patient_trend(df_show: pd.DataFrame, threshold: float, title: str):
    if df_show.empty:
        st.info("No data yet to plot.")
        return
    colors = df_show["status"].map(lambda s: "#d62728" if s == "DIABETIC" else "#2ca02c")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_show["shown_ts"], y=df_show["blood_glucose_level"],
        mode="lines+markers",
        marker=dict(size=8, color=colors, line=dict(width=1, color="black")),
        line=dict(width=2),
        name="Glucose (mg/dL)"
    ))
    fig.add_hrect(y0=70, y1=180, fillcolor="rgba(0,200,0,0.05)", line_width=0)
    fig.add_hline(y=180, line_dash="dash", line_color="orange", annotation_text="180 mg/dL")
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Blood Glucose (mg/dL)", height=420)
    st.plotly_chart(fig, use_container_width=True)

def cohort_tiles(df_log: pd.DataFrame, pids: list, threshold: float):
    cols = st.columns(5)
    for idx, pid in enumerate(pids):
        g = df_log[df_log["patient_id"].astype(str) == str(pid)]
        if g.empty:
            status, prob, last_g, dot, t = "—", "—", "—", "⚪", "—"
        else:
            last = g.sort_values("shown_ts").iloc[-1]
            status = last["status"]
            prob = f"{last['proba']:.2f}"
            last_g = int(last["blood_glucose_level"])
            dot = "🟢" if status == "NORMAL" else "🔴"
            t = pd.to_datetime(last["shown_ts"]).strftime("%d %b %H:%M")
        with cols[idx % 5]:
            st.metric(label=f"{dot} Patient {pid}", value=f"Status: {status}", delta=f"Prob: {prob}")
            st.caption(f"Last: {last_g} mg/dL • {t}")

# ---------- PAGES ----------
def page_all_patients(df, model, feature_cols, threshold):
    st.markdown("## 🏥 All Patients — Command Center")
    pids = get_patient_list(df)
    with st.sidebar:
        st.markdown("### ▶️ Stream Controls")
        autoplay = st.toggle("Auto-play stream (selected patient)", value=st.session_state.autoplay)
        st.session_state.autoplay = autoplay
        refresh = st.slider("Auto-play interval (seconds)", 0.5, 3.0, 1.0, 0.5)
        target_pid = st.selectbox("Select patient to stream", pids, index=0)
        init_patient_pointer(target_pid, df)
        col_a, col_b = st.columns(2)
        if col_a.button("Advance one tick"):
            advance_one_tick(target_pid, df, model, feature_cols, threshold)
            st.rerun()
        if col_b.button("Reset patient stream"):
            st.session_state.stream_index[target_pid] = 0
            st.session_state.log_df = st.session_state.log_df[st.session_state.log_df["patient_id"] != target_pid]
            st.rerun()
    # Clock
    clock_panel(target_pid, df, title="⏱️ Live vs Simulated Time")
    # Autoplay loop
    now = time.time()
    if st.session_state.autoplay and now - st.session_state.last_tick >= refresh:
        did = advance_one_tick(target_pid, df, model, feature_cols, threshold)
        st.session_state.last_tick = now
        if did:
            st.rerun()
    # Cohort
    st.markdown("#### Cohort Overview")
    log = st.session_state.log_df.copy()
    cohort_tiles(log, pids, threshold)
    st.markdown("#### Recent Alerts")
    if log.empty:
        st.info("No alerts yet. Start streaming.")
    else:
        alerts = log[log["proba"] >= threshold].sort_values("shown_ts", ascending=False).head(50)
        st.dataframe(alerts[["patient_id","shown_ts","blood_glucose_level","HbA1c_level","proba","status","advice"]]
                     .rename(columns={"blood_glucose_level":"glucose","HbA1c_level":"hba1c"}),
                     use_container_width=True, hide_index=True)

def page_patient_twin(df, model, feature_cols, threshold):
    st.markdown("## 👤 Patient Twin — Real-Time Cockpit")
    pids = get_patient_list(df)
    with st.sidebar:
        pid = st.selectbox("Patient", pids, index=0)
        init_patient_pointer(pid, df)
        st.markdown("---")
        autoplay = st.toggle("Auto-play (this patient)", value=st.session_state.autoplay, key="autoplay_pt")
        st.session_state.autoplay = autoplay
        refresh = st.slider("Interval (seconds)", 0.5, 3.0, 1.0, 0.5, key="refresh_pt")
        col_a, col_b = st.columns(2)
        if col_a.button("Next tick", key="next_tick_pt"):
            advance_one_tick(pid, df, model, feature_cols, threshold)
            st.rerun()
        if col_b.button("Reset stream", key="reset_tick_pt"):
            st.session_state.stream_index[pid] = 0
            st.session_state.log_df = st.session_state.log_df[st.session_state.log_df["patient_id"] != pid]
            st.rerun()
        st.markdown("---")
        st.markdown("### Threshold")
        st.info(f"Threshold: **{threshold:.3f}** (edit in *Model & Threshold* page)")
    # Clock
    clock_panel(pid, df, title="⏱️ Live vs Simulated Time")
    # Autoplay
    now = time.time()
    if st.session_state.autoplay and now - st.session_state.last_tick >= refresh:
        did = advance_one_tick(pid, df, model, feature_cols, threshold)
        st.session_state.last_tick = now
        if did:
            st.rerun()
    log = st.session_state.log_df.copy()
    g = log[log["patient_id"].astype(str) == str(pid)].sort_values("shown_ts")
    c1, c2, c3, c4 = st.columns(4)
    if g.empty:
        c1.metric("Latest Glucose", "—")
        c2.metric("Latest HbA1c", "—")
        c3.metric("Risk Probability", "—")
        c4.metric("Status", "—")
    else:
        last = g.iloc[-1]
        c1.metric("Latest Glucose", f"{int(last['blood_glucose_level'])} mg/dL")
        c2.metric("Latest HbA1c", f"{float(last['HbA1c_level']):.2f}")
        c3.metric("Risk Probability", f"{float(last['proba']):.3f}")
        c4.metric("Status", last["status"], help=last["advice"])
    plot_patient_trend(g, threshold, f"Patient {pid} — Glucose trend")
    k1, k2, k3, k4 = st.columns(4)
    tir, alerts, mean_glucose, max_prob = compute_kpis(g, threshold)
    k1.metric("Time-in-Range (70–180)%", f"{0 if tir is None else tir:.1f}%")
    k2.metric("Alerts", f"{alerts}")
    k3.metric("Mean Glucose", "—" if np.isnan(mean_glucose) else f"{mean_glucose:.0f} mg/dL")
    k4.metric("Max Prob", "—" if np.isnan(max_prob) else f"{max_prob:.2f}")
    st.markdown("#### Advice Feed")
    if g.empty:
        st.info("No events yet. Start streaming.")
    else:
        show = g[["shown_ts","blood_glucose_level","HbA1c_level","status","advice","proba"]].tail(10)
        show = show.rename(columns={"shown_ts":"time","blood_glucose_level":"glucose","HbA1c_level":"hba1c"})
        st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption("⚠️ Educational prototype — not for medical use.")

def page_model_threshold(df, model, feature_cols, threshold):
    st.markdown("## 🧠 Model & Threshold — Twin Brain")
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            fi = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False).head(15)
            st.plotly_chart(px.bar(fi, x="importance", y="feature", orientation="h", title="Top Features"), use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importances unavailable: {e}")
    st.markdown("### Threshold Tuning — Preview")
    new_thr = st.slider("Threshold", 0.05, 0.95, float(threshold), 0.01)
    try:
        df_eval = df.dropna(subset=feature_cols + ["diabetes"]).copy()
        X_eval, y_eval = df_eval[feature_cols], df_eval["diabetes"].astype(int)
        proba_eval = model.predict_proba(X_eval)[:, 1]
        y_pred_eval = (proba_eval >= new_thr).astype(int)
        acc = accuracy_score(y_eval, y_pred_eval); prec = precision_score(y_eval, y_pred_eval, zero_division=0)
        rec = recall_score(y_eval, y_pred_eval, zero_division=0); f1 = f1_score(y_eval, y_pred_eval, zero_division=0)
        auc = roc_auc_score(y_eval, proba_eval)
        c1, c2, c3, c4, c5 = st.columns(5)
        for c, v, n in zip([c1,c2,c3,c4,c5],[acc,prec,rec,f1,auc],["Accuracy","Precision","Recall","F1","ROC-AUC"]):
            c.metric(n, f"{v:.3f}")
        with st.expander("Confusion Matrix"):
            st.write(pd.DataFrame(confusion_matrix(y_eval, y_pred_eval),
                                  index=["Actual 0 (non-diabetic)", "Actual 1 (diabetic)"],
                                  columns=["Pred 0","Pred 1"]))
    except Exception as e:
        st.warning(f"Evaluation skipped: {e}")
    if st.button("💾 Save threshold"):
        joblib.dump(float(new_thr), "decision_threshold.pkl")
        st.success(f"Saved new threshold {new_thr:.3f} — reload app to use.")

def page_data_logs():
    st.markdown("## 📜 Data & Logs — Twin Memory")
    st.caption("Live predictions are appended to `live_predictions_log.csv` as the stream progresses.")
    st.markdown("### In-session Log (latest 200)")
    if st.session_state.log_df.empty:
        st.info("No log yet in this session.")
    else:
        st.dataframe(st.session_state.log_df.sort_values("shown_ts").tail(200), use_container_width=True, hide_index=True)
    st.markdown("### On-Disk Log")
    if os.path.exists("live_predictions_log.csv"):
        try:
            disk = pd.read_csv("live_predictions_log.csv", parse_dates=["historical_ts","shifted_ts","arrival_ts","shown_ts"])
            st.dataframe(disk.tail(200), use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download full log CSV", data=disk.to_csv(index=False), file_name="live_predictions_log.csv", mime="text/csv")
        except Exception as e:
            st.warning(f"Could not read on-disk log: {e}")
    else:
        st.info("`live_predictions_log.csv` not found yet. Start streaming to create it.")
    if st.button("🗑️ Clear on-disk log"):
        try:
            if os.path.exists("live_predictions_log.csv"):
                os.remove("live_predictions_log.csv")
            st.success("Cleared `live_predictions_log.csv`.")
        except Exception as e:
            st.error(f"Failed to clear: {e}")

# ---------- MAIN ----------
def main():
    st.title("🩺 Digital Twin Framework — Real-Time Diabetes Management")
    st.caption("Prototype for educational decision support — not for medical use.")
    ensure_session()
    try:
        model, feature_cols, threshold = load_model_artifacts()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}"); st.stop()
    try:
        df = load_collected()
    except Exception as e:
        st.error(f"Error loading collected data: {e}"); st.stop()
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Go to", ["All Patients", "Patient Twin", "Model & Threshold", "Data & Logs"], index=0)
    if page == "All Patients":
        page_all_patients(df, model, feature_cols, threshold)
    elif page == "Patient Twin":
        page_patient_twin(df, model, feature_cols, threshold)
    elif page == "Model & Threshold":
        page_model_threshold(df, model, feature_cols, threshold)
    else:
        page_data_logs()

if __name__ == "__main__":
    main()

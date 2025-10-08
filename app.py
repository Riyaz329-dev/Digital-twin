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

# ---------- FORECAST HELPERS ----------
def build_patient_series(df: pd.DataFrame, pid: str) -> pd.DataFrame:
    g = df[df["patient_id"].astype(str) == str(pid)].sort_values("shifted_ts").copy()
    return g[["shifted_ts", "blood_glucose_level", "HbA1c_level"]].rename(
        columns={"shifted_ts":"ts", "blood_glucose_level":"glucose", "HbA1c_level":"hba1c"}
    )

def ewma_forecast(series: pd.Series, steps: int, alpha: float = 0.5, noise_std: float = 3.0) -> np.ndarray:
    if len(series) == 0:
        return np.array([])
    level = series.iloc[-1]
    ew = series.ewm(alpha=alpha).mean().iloc[-1]
    preds = []
    current = level
    for _ in range(steps):
        current = 0.6*current + 0.4*ew + np.random.normal(0, noise_std)
        preds.append(max(current, 0))
    return np.array(preds)

def apply_scenario(glucose_arr: np.ndarray, scenario: str) -> np.ndarray:
    factors = {
        "Maintain (no change)": 1.00,
        "Improve (diet/med −10%)": 0.90,
        "Strong improve (−20%)": 0.80,
        "Worsen (+10%)": 1.10,
    }
    f = factors.get(scenario, 1.00)
    return np.clip(glucose_arr * f, 0, None)

def simulate_future(pid: str, df_src: pd.DataFrame, model, feature_cols: list, threshold: float,
                    horizon_hours: int = 24, freq_minutes: int = 60, method: str = "EWMA",
                    scenario: str = "Maintain (no change)") -> pd.DataFrame:
    hist = build_patient_series(df_src, pid)
    if hist.empty:
        return pd.DataFrame()
    last_time = hist["ts"].max()
    future_index = pd.date_range(
        start=last_time + pd.Timedelta(minutes=freq_minutes),
        periods=max(1, int(horizon_hours*60/freq_minutes)),
        freq=f"{freq_minutes}min"
    )
    if method == "Hold-Last":
        base = np.full(len(future_index), hist["glucose"].iloc[-1])
        noise = np.random.normal(0, 2.0, size=len(base))
        g_fore = np.clip(base + noise, 0, None)
    else:
        g_fore = ewma_forecast(hist["glucose"], steps=len(future_index), alpha=0.5, noise_std=3.0)
    g_fore = apply_scenario(g_fore, scenario)
    hba1c_last = float(hist["hba1c"].iloc[-1])
    h_fore = np.full(len(future_index), hba1c_last)
    last_row = df_src[df_src["patient_id"].astype(str) == str(pid)].sort_values("shifted_ts").iloc[-1].copy()
    fut = pd.DataFrame({
        "patient_id": pid,
        "timestamp": future_index,
        "shifted_ts": future_index,
        "HbA1c_level": h_fore,
        "blood_glucose_level": g_fore
    })
    for col in feature_cols:
        if col in ["HbA1c_level", "blood_glucose_level"]:
            continue
        fut[col] = last_row.get(col, 0)
    X = fut[feature_cols].copy()
    proba = model.predict_proba(X)[:, 1]
    fut["proba"] = proba
    fut["pred"] = (proba >= threshold).astype(int)
    fut["status"] = np.where(fut["pred"] == 1, "DIABETIC", "NORMAL")
    fut["advice"] = [advise(g, h) for g, h in zip(fut["blood_glucose_level"], fut["HbA1c_level"])]
    fut["is_forecast"] = True
    return fut

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

    # Sidebar controls
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

    # Clock panel
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

    # Alerts
    st.markdown("#### Recent Alerts")
    if log.empty:
        st.info("No alerts yet. Start streaming.")
    else:
        alerts = log[log["proba"] >= threshold].sort_values("shown_ts", ascending=False).head(50)
        st.dataframe(
            alerts[["patient_id","shown_ts","blood_glucose_level","HbA1c_level","proba","status","advice"]]
            .rename(columns={"blood_glucose_level":"glucose","HbA1c_level":"hba1c"}),
            use_container_width=True, hide_index=True
        )

def page_patient_twin(df, model, feature_cols, threshold):
    st.markdown("## 👤 Patient Twin — Real-Time Cockpit")

    pids = get_patient_list(df)
    with st.sidebar:
        pid = st.selectbox("Patient", pids, index=0)
        init_patient_pointer(pid, df)
        st.markdown("---")
        st.markdown("### ▶️ Live Stream")
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
        st.caption("Current decision threshold used for status.")
        st.info(f"Threshold: **{threshold:.3f}** (edit in *Model & Threshold* page)")

    # Clock panel
    clock_panel(pid, df, title="⏱️ Live vs Simulated Time")

    # Autoplay step
    now = time.time()
    if st.session_state.autoplay and now - st.session_state.last_tick >= refresh:
        did = advance_one_tick(pid, df, model, feature_cols, threshold)
        st.session_state.last_tick = now
        if did:
            st.rerun()

    # Current log slice for patient
    log = st.session_state.log_df.copy()
    g = log[log["patient_id"].astype(str) == str(pid)].sort_values("shown_ts")

    # Header cards: current twin state
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

    # Historical trend
    plot_patient_trend(g, threshold, f"Patient {pid} — Glucose trend (green=NORMAL, red=DIABETIC)")

    # KPIs + Advice + Events
    k1, k2, k3, k4 = st.columns(4)
    tir, alerts, mean_glucose, max_prob = compute_kpis(g, threshold)
    k1.metric("Time-in-Range (70–180) %", f"{0 if tir is None else tir:.1f}%")
    k2.metric("Alerts in Window", f"{alerts}")
    k3.metric("Mean Glucose", "—" if np.isnan(mean_glucose) else f"{mean_glucose:.0f} mg/dL")
    k4.metric("Max Probability", "—" if np.isnan(max_prob) else f"{max_prob:.2f}")

    st.markdown("#### Advice Feed")
    if g.empty:
        st.info("No events yet. Start streaming.")
    else:
        show = g[["shown_ts","blood_glucose_level","HbA1c_level","status","advice","proba"]].tail(10)
        show = show.rename(columns={
            "shown_ts":"time",
            "blood_glucose_level":"glucose",
            "HbA1c_level":"hba1c"
        })
        st.dataframe(show, use_container_width=True, hide_index=True)

    # ---------- 🔮 FUTURE SIMULATOR ----------
    st.markdown("---")
    st.markdown("### 🔮 Future Simulator (Forecast)")
    col_fs1, col_fs2, col_fs3, col_fs4 = st.columns(4)
    horizon = col_fs1.slider("Horizon (hours)", 1, 72, 24, 1)
    freq = col_fs2.selectbox("Frequency", [15, 30, 60, 120], index=2, format_func=lambda m: f"{m} min")
    method = col_fs3.selectbox("Method", ["EWMA", "Hold-Last"], index=0)
    scenario = col_fs4.selectbox("Scenario", [
        "Maintain (no change)",
        "Improve (diet/med −10%)",
        "Strong improve (−20%)",
        "Worsen (+10%)",
    ], index=0)

    if st.button("Generate forecast"):
        fut = simulate_future(pid, df, model, feature_cols, threshold,
                              horizon_hours=horizon, freq_minutes=freq,
                              method=method, scenario=scenario)
        if fut.empty:
            st.info("No history for this patient to forecast from.")
        else:
            # Small future table
            st.dataframe(
                fut[["shifted_ts","blood_glucose_level","HbA1c_level","proba","status","advice"]]
                .rename(columns={"shifted_ts":"time","blood_glucose_level":"glucose","HbA1c_level":"hba1c"})
                .head(20),
                use_container_width=True, hide_index=True
            )

            # Overlay forecast on the chart
            base = g.copy()
            base["is_forecast"] = False
            plot_df = pd.concat([base, fut], ignore_index=True)

            fig = go.Figure()
            hist_df = plot_df[plot_df["is_forecast"] == False]
            if not hist_df.empty:
                fig.add_trace(go.Scatter(
                    x=hist_df["shown_ts"], y=hist_df["blood_glucose_level"],
                    mode="lines+markers", name="Historical",
                    line=dict(width=2), marker=dict(size=7, line=dict(width=1, color="black"))
                ))
            fut_df = plot_df[plot_df["is_forecast"] == True]
            if not fut_df.empty:
                fig.add_trace(go.Scatter(
                    x=fut_df["shifted_ts"], y=fut_df["blood_glucose_level"],
                    mode="lines+markers", name="Forecast",
                    line=dict(width=2, dash="dash"), marker=dict(size=7)
                ))
            fig.add_hrect(y0=70, y1=180, fillcolor="rgba(0,200,0,0.05)", line_width=0)
            fig.add_hline(y=180, line_dash="dash", line_color="orange", annotation_text="180 mg/dL")
            fig.update_layout(
                title=f"Patient {pid} — Historical vs Forecasted Glucose",
                xaxis_title="Time", yaxis_title="Blood Glucose (mg/dL)", height=420
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast alerts
            alerts_fut = fut[fut["proba"] >= threshold][["shifted_ts","blood_glucose_level","HbA1c_level","proba","status","advice"]]
            if alerts_fut.empty:
                st.success("No forecasted alerts in the selected horizon.")
            else:
                st.warning(f"{len(alerts_fut)} forecasted alert(s) in the selected horizon.")
                st.dataframe(
                    alerts_fut.rename(columns={"shifted_ts":"time","blood_glucose_level":"glucose","HbA1c_level":"hba1c"}),
                    use_container_width=True, hide_index=True
                )

    st.caption("⚠️ Educational prototype — not for medical use. Forecasts are scenario simulations.")

def page_model_threshold(df, model, feature_cols, threshold):
    st.markdown("## 🧠 Model & Threshold — Twin Brain")

    st.markdown("### Global Feature Importance")
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            fi = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False).head(15)
            st.plotly_chart(px.bar(fi, x="importance", y="feature", orientation="h", title="Top Features"), use_container_width=True)
        else:
            st.info("Model does not expose `feature_importances_`.")
    except Exception as e:
        st.warning(f"Feature importances unavailable: {e}")

    st.markdown("### Threshold Tuning — Preview on Collected Data")
    new_thr = st.slider("Threshold", 0.05, 0.95, float(threshold), 0.01)

    try:
        df_eval = df.dropna(subset=feature_cols + ["diabetes"]).copy()
        X_eval, y_eval = df_eval[feature_cols], df_eval["diabetes"].astype(int)
        proba_eval = model.predict_proba(X_eval)[:, 1]
        y_pred_eval = (proba_eval >= new_thr).astype(int)

        acc  = accuracy_score(y_eval, y_pred_eval)
        prec = precision_score(y_eval, y_pred_eval, zero_division=0)
        rec  = recall_score(y_eval, y_pred_eval, zero_division=0)
        f1   = f1_score(y_eval, y_pred_eval, zero_division=0)
        auc  = roc_auc_score(y_eval, proba_eval)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1-score", f"{f1:.3f}")
        c5.metric("ROC-AUC", f"{auc:.3f}")

        with st.expander("Confusion Matrix"):
            st.write(pd.DataFrame(confusion_matrix(y_eval, y_pred_eval),
                                  index=["Actual 0 (non-diabetic)", "Actual 1 (diabetic)"],
                                  columns=["Pred 0", "Pred 1"]))
    except Exception as e:
        st.warning(f"Could not evaluate on collected data: {e}")

    if st.button("💾 Save this threshold"):
        try:
            joblib.dump(float(new_thr), "decision_threshold.pkl")
            st.success(f"Saved threshold = {new_thr:.3f}. Reload the app to use it.")
        except Exception as e:
            st.error(f"Failed to save threshold: {e}")

def page_data_logs():
    st.markdown("## 📜 Data & Logs — Twin Memory")
    st.caption("Live predictions are appended to `live_predictions_log.csv` as the stream progresses.")

    # In-session log
    st.markdown("### In-session Log (latest 200)")
    if st.session_state.log_df.empty:
        st.info("No log yet in this session.")
    else:
        st.dataframe(st.session_state.log_df.sort_values("shown_ts").tail(200), use_container_width=True, hide_index=True)

    # On-disk log
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

    # Clear disk log
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

    # Load artifacts and data
    try:
        model, feature_cols, threshold = load_model_artifacts()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

    try:
        df = load_collected()
    except Exception as e:
        st.error(f"Error loading collected data: {e}")
        st.stop()

    # Sidebar navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["All Patients", "Patient Twin", "Model & Threshold", "Data & Logs"],
        index=0
    )

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

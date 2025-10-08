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
"""Streamlit application for the Digital Twin diabetes management demo.

The repository bundles the model artifacts (``best_model.pkl`` and feature
columns), a tuned classification threshold, and a cleaned historical dataset.

Typical usage::

    pip install -r requirements.txt
    streamlit run app.py

This module intentionally keeps module level statements unindented so that
copy-pasting the file into environments such as the GitHub editor does not
introduce accidental leading whitespace that could trigger ``IndentationError``
on load.
"""

import os
import time
from pathlib import Path
from typing import Optional

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
@@ -54,53 +60,56 @@ def load_collected() -> pd.DataFrame:
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

    proba_series = g["proba"] if "proba" in g else pd.Series(dtype=float)
    alerts = int((proba_series >= threshold).sum())

    mean_glucose = float(g["blood_glucose_level"].mean()) if not g.empty else np.nan
    max_prob = float(g.get("proba", pd.Series([0.0])).max()) if "proba" in g else np.nan
    max_prob = float(proba_series.max()) if not proba_series.empty else np.nan
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
@@ -144,90 +153,107 @@ def clock_panel(selected_pid: str | None, df: pd.DataFrame, title: str = "⏱️
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
def ewma_forecast(series: pd.Series, steps: int, alpha: float = 0.5, noise_std: float = 3.0,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if len(series) == 0:
        return np.array([])
    rng = rng or np.random.default_rng()
    level = series.iloc[-1]
    ew = series.ewm(alpha=alpha).mean().iloc[-1]
    preds = []
    current = level
    for _ in range(steps):
        current = 0.6*current + 0.4*ew + np.random.normal(0, noise_std)
        current = 0.6*current + 0.4*ew + rng.normal(0, noise_std)
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

def summarize_forecast(fut: pd.DataFrame, threshold: float) -> dict:
    alerts = fut[fut["proba"] >= threshold]
    first_alert_time = alerts["shifted_ts"].min() if not alerts.empty else None
    tir = float((fut["blood_glucose_level"].between(70, 180).mean() * 100.0)) if not fut.empty else np.nan
    summary = {
        "mean_glucose": float(fut["blood_glucose_level"].mean()) if not fut.empty else np.nan,
        "min_glucose": float(fut["blood_glucose_level"].min()) if not fut.empty else np.nan,
        "max_glucose": float(fut["blood_glucose_level"].max()) if not fut.empty else np.nan,
        "alerts_count": int(len(alerts)),
        "first_alert_time": first_alert_time,
        "tir": tir,
    }
    return summary

def simulate_future(pid: str, df_src: pd.DataFrame, model, feature_cols: list, threshold: float,
                    horizon_hours: int = 24, freq_minutes: int = 60, method: str = "EWMA",
                    scenario: str = "Maintain (no change)") -> pd.DataFrame:
                    scenario: str = "Maintain (no change)", random_seed: Optional[int] = None) -> pd.DataFrame:
    hist = build_patient_series(df_src, pid)
    if hist.empty:
        return pd.DataFrame()
    last_time = hist["ts"].max()
    rng = np.random.default_rng(random_seed)
    future_index = pd.date_range(
        start=last_time + pd.Timedelta(minutes=freq_minutes),
        periods=max(1, int(horizon_hours*60/freq_minutes)),
        freq=f"{freq_minutes}min"
    )
    if method == "Hold-Last":
        base = np.full(len(future_index), hist["glucose"].iloc[-1])
        noise = np.random.normal(0, 2.0, size=len(base))
        noise = rng.normal(0, 2.0, size=len(base))
        g_fore = np.clip(base + noise, 0, None)
    else:
        g_fore = ewma_forecast(hist["glucose"], steps=len(future_index), alpha=0.5, noise_std=3.0)
        g_fore = ewma_forecast(hist["glucose"], steps=len(future_index), alpha=0.5, noise_std=3.0, rng=rng)
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
@@ -385,57 +411,99 @@ def page_patient_twin(df, model, feature_cols, threshold):
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

    with st.expander("How should I interpret this forecast?", expanded=False):
        st.markdown(
            "- **EWMA** smooths recent glucose values and extends them forward with light noise to mimic natural variation.\n"
            "- **Hold-Last** keeps the latest reading and adds minimal noise — best for a conservative outlook.\n"
            "- **Scenarios** scale the glucose curve to explore lifestyle or medication changes.\n"
            "- Forecast probabilities use the loaded model on simulated future vitals; treat them as directional guidance, not certainties."
        )

    col_seed1, col_seed2 = st.columns([1, 1])
    lock_seed = col_seed1.checkbox(
        "Lock forecast randomness", value=st.session_state.get("forecast_lock", False)
    )
    seed_value: Optional[int] = None
    if lock_seed:
        default_seed = int(st.session_state.get("forecast_seed", 0))
        seed_input = col_seed2.number_input(
            "Random seed", min_value=0, max_value=9999, value=default_seed, step=1, format="%d"
        )
        seed_value = int(seed_input)
        st.session_state.forecast_seed = seed_value
    st.session_state.forecast_lock = lock_seed

    if st.button("Generate forecast"):
        fut = simulate_future(pid, df, model, feature_cols, threshold,
                              horizon_hours=horizon, freq_minutes=freq,
                              method=method, scenario=scenario)
                              method=method, scenario=scenario, random_seed=seed_value)
        if fut.empty:
            st.info("No history for this patient to forecast from.")
        else:
            summary = summarize_forecast(fut, threshold)
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            col_met1.metric(
                "Avg glucose", "—" if np.isnan(summary["mean_glucose"]) else f"{summary['mean_glucose']:.0f} mg/dL"
            )
            if np.isnan(summary["min_glucose"]) or np.isnan(summary["max_glucose"]):
                min_max_label = "—"
            else:
                min_max_label = f"{summary['min_glucose']:.0f} – {summary['max_glucose']:.0f} mg/dL"
            col_met2.metric("Min → Max", min_max_label)
            col_met3.metric("Forecast alerts", f"{summary['alerts_count']}")
            if summary["first_alert_time"] is None:
                first_alert_label = "None in horizon"
            else:
                first_alert_label = pd.to_datetime(summary["first_alert_time"]).strftime("%d %b %H:%M")
            col_met4.metric("First alert ETA", first_alert_label)

            tir_value = 0.0 if np.isnan(summary["tir"]) else summary["tir"]
            st.caption(f"Scenario applied: **{scenario}** • Time-in-range ≈ {tir_value:.1f}%")

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
@@ -523,64 +591,85 @@ def page_data_logs():
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

def page_source_code():
    st.markdown("## 💻 Application Source")
    st.caption("Preview or download the full Streamlit app code directly from within the app.")

    try:
        code_path = Path(__file__).resolve()
        code_text = code_path.read_text(encoding="utf-8")
    except Exception as e:
        st.error(f"Unable to read source file: {e}")
        return

    st.download_button(
        "⬇️ Download app.py",
        data=code_text,
        file_name="app.py",
        mime="text/x-python",
    )
    st.code(code_text, language="python")

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
        ["All Patients", "Patient Twin", "Model & Threshold", "Data & Logs", "Source Code"],
        index=0
    )

    if page == "All Patients":
        page_all_patients(df, model, feature_cols, threshold)
    elif page == "Patient Twin":
        page_patient_twin(df, model, feature_cols, threshold)
    elif page == "Model & Threshold":
        page_model_threshold(df, model, feature_cols, threshold)
    else:
    elif page == "Data & Logs":
        page_data_logs()
    else:
        page_source_code()

if __name__ == "__main__":
    main()

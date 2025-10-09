# -*- coding: utf-8 -*-
# app.py — Digital Twin Framework for Real-Time Diabetes Management

import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from datetime import datetime as _dt
from zoneinfo import ZoneInfo  # UK time support

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

# Optional ARIMA for forecasting
try:
    from statsmodels.tsa.arima.model import ARIMA
    _ARIMA_OK = True
except Exception:
    _ARIMA_OK = False

# ---------- GLOBAL CONFIG ----------
st.set_page_config(page_title="Digital Twin: Real-Time Diabetes Management",
                   page_icon="🩺", layout="wide")

# ---------- TIME HELPERS (UK) ----------
UK_TZ = ZoneInfo("Europe/London")

def now_uk():
    """Timezone-aware UK 'now'."""
    return _dt.now(UK_TZ)

# ---------- FILE NAMES ----------
ART_MODEL = "best_model.pkl"
ART_THRESHOLD = "decision_threshold.pkl"
ART_FEATURES = "feature_cols.pkl"
COLLECTED_PATHS = ["Collected_cleaned.csv", "Collected_dataset.csv", "Collected.csv"]
KAGGLE_PATHS = ["Dataset.csv", "Kaggle.csv", "kaggle_diabetes.csv"]
LOG_PATH = "live_predictions_log.csv"

# ---------- IO HELPERS ----------
@st.cache_data(show_spinner=False)
def read_csv_any(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return pd.DataFrame()

def append_to_csv(path: str, df_append: pd.DataFrame):
    header = not os.path.exists(path)
    df_append.to_csv(path, mode="a", header=header, index=False)

def safe_to_datetime(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT

def ensure_columns(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ---------- DATA LOADING ----------
@st.cache_data(show_spinner=False)
def load_collected():
    df = read_csv_any(COLLECTED_PATHS)
    if df.empty:
        return df

    needed = ["patient_id", "timestamp", "blood_glucose_level", "HbA1c_level"]
    df = ensure_columns(df, needed)

    df["timestamp"] = df["timestamp"].apply(safe_to_datetime)
    # Shift historical timeline to align with today (simulation)
    try:
        first_ts = df["timestamp"].min()
        today0 = pd.Timestamp(now_uk().date(), tz=UK_TZ)
        delta = today0 - pd.Timestamp(first_ts.date(), tz=UK_TZ)
        df["shifted_ts"] = df["timestamp"].dt.tz_localize(UK_TZ, nonexistent="shift_forward", ambiguous="NaT") + delta
        df["shifted_ts"] = df["shifted_ts"].dt.tz_convert(UK_TZ)
    except Exception:
        df["shifted_ts"] = df["timestamp"]

    df["blood_glucose_level"] = pd.to_numeric(df["blood_glucose_level"], errors="coerce")
    df["HbA1c_level"] = pd.to_numeric(df["HbA1c_level"], errors="coerce")

    df = df.sort_values(["patient_id", "shifted_ts"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_kaggle():
    return read_csv_any(KAGGLE_PATHS)

# ---------- PREPROCESSOR / MODEL ----------
def build_preprocessor(train_df: pd.DataFrame, y_col="diabetes"):
    num_cols, cat_cols = [], []
    for c in train_df.columns:
        if c == y_col:
            continue
        if train_df[c].dtype.kind in "biufc":
            num_cols.append(c)
        else:
            cat_cols.append(c)
    drop_cols = [c for c in ["patient_id", "timestamp", "shifted_ts"] if c in train_df.columns]
    num_cols = [c for c in num_cols if c not in drop_cols]
    cat_cols = [c for c in cat_cols if c not in drop_cols]

    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return pre, num_cols, cat_cols

def pick_threshold(y_true, y_prob):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-9)
    ix = int(np.nanargmax(f1)) if len(f1) else 0
    return float(thr[ix]) if len(thr) else 0.5

@st.cache_resource(show_spinner=True)
def load_or_train_model():
    if os.path.exists(ART_MODEL) and os.path.exists(ART_THRESHOLD) and os.path.exists(ART_FEATURES):
        model = joblib.load(ART_MODEL)
        threshold = float(joblib.load(ART_THRESHOLD))
        feature_cols = list(joblib.load(ART_FEATURES))
        meta = {"trained": False}
        return model, threshold, feature_cols, meta

    df = load_kaggle()
    if df.empty or "diabetes" not in df.columns:
        # Tiny fallback training (if dataset missing)
        X = pd.DataFrame({"HbA1c_level": [5.5, 8.0, 6.2, 7.4],
                          "blood_glucose_level": [100, 240, 150, 200]})
        y = np.array([0, 1, 0, 1])
        pre = ColumnTransformer([("num", "passthrough", list(X.columns))])
        base = GradientBoostingClassifier(random_state=42)
        model = Pipeline([("pre", pre),
                          ("clf", CalibratedClassifierCV(base, cv=3))]).fit(X, y)
        return model, 0.5, list(X.columns), {"trained": True, "auc": None}

    df = df.copy()
    df["diabetes"] = pd.to_numeric(df["diabetes"], errors="coerce").fillna(0).astype(int)

    preferred = ["age", "hypertension", "heart_disease", "bmi",
                 "HbA1c_level", "blood_glucose_level", "gender", "smoking_history"]
    kept = [c for c in preferred if c in df.columns] or [c for c in df.columns if c != "diabetes"]

    X = df[kept].copy()
    y = df["diabetes"].values

    pre, _, _ = build_preprocessor(pd.concat([X, df["diabetes"]], axis=1))
    base = GradientBoostingClassifier(random_state=42)
    model = Pipeline([("pre", pre),
                      ("clf", CalibratedClassifierCV(base, cv=3))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(Xtr, ytr)
    yprob = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, yprob)
    thr = pick_threshold(yte, yprob)

    # Persist artifacts
    try:
        joblib.dump(model, ART_MODEL)
        joblib.dump(thr, ART_THRESHOLD)
        joblib.dump(kept, ART_FEATURES)
    except Exception:
        pass

    return model, float(thr), kept, {"trained": True, "auc": float(auc)}

def predict_row(model, feature_cols, row: pd.Series) -> float:
    x = pd.DataFrame([row.get(c, np.nan) for c in feature_cols], index=feature_cols).T
    try:
        return float(model.predict_proba(x)[:, 1][0])
    except Exception:
        return 0.0

# ---------- CORE UTILS ----------
def advise(glucose: float, hba1c: float) -> str:
    if glucose < 70:
        return "🟠 Low: Treat hypoglycaemia per your care plan (15–20g fast carbs, recheck in 15 min)."
    if glucose < 140 and hba1c < 6:
        return "🟢 Normal: Maintain healthy diet and regular exercise."
    if 140 <= glucose < 180 or (6 <= hba1c < 7):
        return "⚠️ Slightly elevated: Recheck in a few hours and limit sugar intake."
    if glucose >= 180 or hba1c >= 7:
        return "🔴 High: Monitor closely and consult your clinician if persistent."
    return "ℹ️ Unable to determine — please verify readings."

def compute_kpis(g: pd.DataFrame, threshold: float):
    tir = None
    if not g.empty and "blood_glucose_level" in g:
        # EDIT: TIR uses 70–180 mg/dL (inclusive)
        tir = float(((g["blood_glucose_level"] >= 70) & (g["blood_glucose_level"] <= 180)).mean() * 100.0)
    alerts = int((g.get("proba", pd.Series([])) >= threshold).sum()) if "proba" in g else 0
    mean_glucose = float(g["blood_glucose_level"].mean()) if not g.empty else np.nan
    max_prob = float(g.get("proba", pd.Series([0.0])).max()) if "proba" in g else np.nan
    return tir, alerts, mean_glucose, max_prob

def forecast_hold_last(series: pd.Series, steps: int = 6):
    if series.empty:
        return []
    last = float(series.iloc[-1])
    return [last] * steps

def forecast_ewma(series: pd.Series, steps: int = 6, alpha: float = 0.3):
    if series.empty:
        return []
    level = float(series.iloc[0])
    for v in series.iloc[1:]:
        level = alpha * float(v) + (1 - alpha) * level
    return [level] + [level] * (steps - 1)

def forecast_arima(series: pd.Series, steps: int = 6):
    if not _ARIMA_OK or len(series) < 8:
        return forecast_ewma(series, steps=steps, alpha=0.4)
    try:
        model = ARIMA(series.astype(float), order=(1, 1, 1))
        fit = model.fit()
        fc = fit.forecast(steps=steps)
        return [float(x) for x in fc.values]
    except Exception:
        return forecast_ewma(series, steps=steps, alpha=0.4)

# ---------- STATE ----------
def init_state():
    if "log_df" not in st.session_state:
        st.session_state.log_df = pd.DataFrame(columns=[
            "patient_id", "historical_ts", "shifted_ts", "arrival_ts", "shown_ts",
            "blood_glucose_level", "HbA1c_level", "proba", "pred", "status", "advice"
        ])
    if "ptr" not in st.session_state:
        st.session_state.ptr = {}

# ---------- PAGES ----------
def page_overview(collected, model, threshold):
    st.subheader("🗺️ Command Center")
    if collected.empty:
        st.info("No collected dataset found. Place `Collected_cleaned.csv` or `Collected_dataset.csv` in the app folder.")
        return

    pats = sorted(collected["patient_id"].dropna().unique().tolist())
    sel = st.multiselect("Select patients", pats, default=pats[: min(5, len(pats))])
    if not sel:
        st.warning("Please pick at least one patient.")
        return

    g = st.session_state.log_df
    gsel = g[g["patient_id"].isin(sel)] if not g.empty else collected[collected["patient_id"].isin(sel)]

    tir, alerts, mean_glucose, max_prob = compute_kpis(gsel, threshold)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time in Range (70–180)", f"{tir:.1f}%" if tir is not None else "—")
    c2.metric("Alerts (≥thr)", f"{alerts}")
    c3.metric("Mean Glucose", f"{mean_glucose:.0f} mg/dL" if not np.isnan(mean_glucose) else "—")
    c4.metric("Max Risk Prob", f"{max_prob:.2f}" if not np.isnan(max_prob) else "—")

    plot_df = collected[collected["patient_id"].isin(sel)].copy()
    plot_df = plot_df.sort_values("shifted_ts")
    st.line_chart(plot_df.set_index("shifted_ts")[["blood_glucose_level"]], height=260)

    st.caption("This is a simulated real-time replay aligned to Europe/London. Forecasts are future projections. Not a medical device.")

def add_next_tick(pid, dfp, model, threshold, feature_cols):
    ptr = st.session_state.ptr.get(pid, 0)
    if ptr >= len(dfp):
        st.warning("No more historical ticks for this patient.")
        return
    row = dfp.iloc[ptr]
    st.session_state.ptr[pid] = ptr + 1

    feat = pd.Series({
        "patient_id": row.get("patient_id"),
        "blood_glucose_level": float(row.get("blood_glucose_level", np.nan)),
        "HbA1c_level": float(row.get("HbA1c_level", np.nan)),
    })
    proba = predict_row(model, feature_cols, feat)
    pred = int(proba >= threshold)
    status = "DIABETIC" if pred == 1 else "NORMAL"

    rec = {
        "patient_id": row.get("patient_id"),
        "historical_ts": row.get("timestamp", row.get("shifted_ts")),
        "shifted_ts": row.get("shifted_ts"),
        "arrival_ts": now_uk(),
        "shown_ts": row.get("shifted_ts"),
        "blood_glucose_level": float(row.get("blood_glucose_level", np.nan)),
        "HbA1c_level": float(row.get("HbA1c_level", np.nan)),
        "proba": proba,
        "pred": pred,
        "status": status,
        "advice": advise(float(row.get("blood_glucose_level", np.nan)),
                         float(row.get("HbA1c_level", np.nan))),
    }

    st.session_state.log_df = pd.concat([st.session_state.log_df, pd.DataFrame([rec])], ignore_index=True)
    try:
        append_to_csv(LOG_PATH, pd.DataFrame([rec]))
    except Exception:
        pass

def page_patient_twin(collected, model, threshold, feature_cols):
    st.subheader("🧍 Patient Twin")
    if collected.empty:
        st.info("No collected dataset found.")
        return

    pats = sorted(collected["patient_id"].dropna().unique().tolist())
    pid = st.selectbox("Patient", pats)

    dfp = collected[collected["patient_id"] == pid].sort_values("shifted_ts").reset_index(drop=True)
    if pid not in st.session_state.ptr:
        st.session_state.ptr[pid] = 0

    with st.sidebar:
        st.markdown("### ▶️ Stream Controls")
        c1, c2 = st.columns(2)
        if c1.button("Next tick"):
            add_next_tick(pid, dfp, model, threshold, feature_cols)
            st.rerun()
        if c2.button("Reset stream"):
            st.session_state.ptr[pid] = 0
            st.session_state.log_df = st.session_state.log_df[st.session_state.log_df["patient_id"] != pid].copy()
            st.experimental_rerun()

    lg = st.session_state.log_df[st.session_state.log_df["patient_id"] == pid].copy()
    tir, alerts, mean_glucose, max_prob = compute_kpis(lg, threshold)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TIR (70–180)", f"{tir:.1f}%" if tir is not None else "—")
    c2.metric("Alerts (≥thr)", f"{alerts}")
    c3.metric("Mean Glucose", f"{mean_glucose:.0f} mg/dL" if not np.isnan(mean_glucose) else "—")
    c4.metric("Max Risk Prob", f"{max_prob:.2f}" if not np.isnan(max_prob) else "—")

    if not lg.empty:
        disp = lg.sort_values("shown_ts").copy()
        st.line_chart(disp.set_index("shown_ts")[["blood_glucose_level"]], height=260)
        # Status strip for last N
        lastn = lg.tail(6).copy()
        cols = st.columns(len(lastn))
        for idx, (_, last) in enumerate(lastn.iterrows()):
            last_status = last["status"]; prob = f"{last['proba']:.2f}"
            last_g = int(last["blood_glucose_level"]); dot = "🟢" if last_status=="NORMAL" else "🔴"
            t = pd.to_datetime(last["shown_ts"]).strftime("%d %b %H:%M")
            with cols[idx]:
                st.markdown(f"**{dot} {t}**")
                st.caption(f"{last_g} mg/dL · p={prob}")
                st.caption(last["advice"])

    # Forecast (simple what-if)
    st.markdown("### 🔮 Short-term Forecast (What-if)")
    horizon = st.slider("Steps (30–60 min steps)", min_value=4, max_value=12, value=6, step=1)
    method = st.selectbox("Method", ["Hold-Last", "EWMA", "ARIMA (if available)"])
    alpha = st.slider("EWMA α (if EWMA)", 0.1, 0.9, 0.3, 0.1)
    last_series = lg["blood_glucose_level"] if not lg.empty else dfp["blood_glucose_level"]

    if method == "Hold-Last":
        fc = forecast_hold_last(last_series, steps=horizon)
    elif method == "EWMA":
        fc = forecast_ewma(last_series, steps=horizon, alpha=alpha)
    else:
        fc = forecast_arima(last_series, steps=horizon)

    fc_df = pd.DataFrame({"step": np.arange(1, len(fc)+1), "forecast_glucose": fc})
    st.dataframe(fc_df, use_container_width=True)
    st.line_chart(fc_df.set_index("step"), height=220)

    st.info("Educational prototype — advisory text only. **Not a medical device.**")

def page_model_info(meta, threshold):
    st.subheader("🧠 Model & Threshold")
    st.write("Classifier: Gradient Boosting (calibrated) or loaded artifact.")
    if meta.get("auc") is not None:
        st.metric("Validation AUC", f"{meta['auc']:.3f}")
    else:
        st.caption("AUC not available (loaded artifacts or fallback).")
    st.metric("Operating Threshold", f"{threshold:.3f}")
    st.caption("Threshold chosen by maximizing F1 on validation PR curve; persisted as an artifact.")

def page_logs():
    st.subheader("📜 Data & Logs")
    lg = st.session_state.log_df.copy()
    if lg.empty:
        st.info("Log is empty. Use Patient Twin → Next tick.")
        return
    st.dataframe(lg.sort_values("shown_ts"), use_container_width=True)
    csv = lg.to_csv(index=False).encode("utf-8")
    st.download_button("Download log CSV", data=csv, file_name="live_predictions_log.csv", mime="text/csv")

# ---------- MAIN ----------
def main():
    st.title("Digital Twin Framework — Real-Time Diabetes (Simulated)")
    st.markdown(
        "> This demo replays collected data on a real-time clock and supports short-term forecasting "
        "(Hold-Last, EWMA, ARIMA). Swap the CSV stream with a device/API to make it fully live."
    )

    init_state()

    collected = load_collected()
    model, threshold, feature_cols, meta = load_or_train_model()

    tab1, tab2, tab3, tab4 = st.tabs(["Command Center", "Patient Twin", "Model & Threshold", "Data & Logs"])
    with tab1:
        page_overview(collected, model, threshold)
    with tab2:
        page_patient_twin(collected, model, threshold, feature_cols)
    with tab3:
        page_model_info(meta, threshold)
    with tab4:
        page_logs()

    st.caption("© 2025 Educational prototype. Not a medical device.")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# app.py — Digital Twin Framework for Real-Time Diabetes Management (Streamlit)

import os
import io
import time
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Optional ARIMA for forecasting
try:
    from statsmodels.tsa.arima.model import ARIMA
    _HAS_ARIMA = True
except Exception:
    _HAS_ARIMA = False

# --------- Helpers ---------

def now_uk() -> datetime:
    """Current time in Europe/London (naive, for CSV compatibility)."""
    # Streamlit Cloud often doesn't have zoneinfo; keep it simple and naive
    return datetime.utcnow() + timedelta(hours=1)  # UK (BST) approx; your dataset is shifted anyway

@st.cache_data(show_spinner=False)
def read_csv_any(paths: List[str]) -> pd.DataFrame:
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            return df
    return pd.DataFrame()

def safe_to_datetime(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def append_to_csv(path: str, df_append: pd.DataFrame):
    header = not os.path.exists(path)
    df_append.to_csv(path, mode="a", header=header, index=False)

# --------- Data loading & model ---------

ART_MODEL = "best_model.pkl"
ART_THRESHOLD = "decision_threshold.pkl"
ART_FEATURES = "feature_cols.pkl"

@st.cache_data(show_spinner=False)
def load_collected() -> pd.DataFrame:
    # Try multiple names users shared
    df = read_csv_any(["Collected_cleaned.csv", "Collected_dataset.csv", "Collected.csv"])
    if df.empty:
        return df

    # columns typically present in your dataset:
    needed = ["patient_id","timestamp","blood_glucose_level","HbA1c_level"]
    df = ensure_columns(df, needed)

    # timestamp -> datetime
    if "timestamp" in df:
        df["timestamp"] = df["timestamp"].apply(safe_to_datetime)
    else:
        df["timestamp"] = pd.to_datetime("now")

    # Shift to "today UK" so the stream feels real-time (simulation)
    # Compute the delta between first ts and today 00:00
    try:
        first_ts = df["timestamp"].min()
        today0 = pd.Timestamp(now_uk().date())
        delta = today0 - pd.Timestamp(first_ts.date())
        df["shifted_ts"] = df["timestamp"] + delta
    except Exception:
        df["shifted_ts"] = df["timestamp"]

    # Basic types
    if "blood_glucose_level" in df:
        df["blood_glucose_level"] = pd.to_numeric(df["blood_glucose_level"], errors="coerce")
    if "HbA1c_level" in df:
        df["HbA1c_level"] = pd.to_numeric(df["HbA1c_level"], errors="coerce")

    # Order
    df = df.sort_values(["patient_id","shifted_ts"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_kaggle() -> pd.DataFrame:
    # User provided Dataset.csv path
    df = read_csv_any(["Dataset.csv", "Kaggle.csv", "kaggle_diabetes.csv"])
    return df

def build_preprocessor(train_df: pd.DataFrame, y_col: str = "diabetes"):
    # Identify types
    num_cols = []
    cat_cols = []
    for c in train_df.columns:
        if c == y_col:
            continue
        if train_df[c].dtype.kind in "biufc":
            num_cols.append(c)
        else:
            # object-like -> cat
            cat_cols.append(c)
    # Common identifiers that shouldn't be used
    drop_cols = [c for c in ["patient_id","timestamp","shifted_ts"] if c in num_cols + cat_cols]
    num_cols = [c for c in num_cols if c not in drop_cols]
    cat_cols = [c for c in cat_cols if c not in drop_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return pre, num_cols, cat_cols

def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray, target="f1") -> float:
    # Choose threshold by max F1 on PR curve
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-9)
    ix = np.nanargmax(f1)
    # Map f1 index to threshold index
    best_thr = thr[ix] if len(thr) else 0.5
    return float(best_thr)

@st.cache_resource(show_spinner=True)
def load_or_train_model() -> Tuple[Pipeline, float, List[str], Dict]:
    # Try artifacts
    if os.path.exists(ART_MODEL) and os.path.exists(ART_THRESHOLD) and os.path.exists(ART_FEATURES):
        model = joblib.load(ART_MODEL)
        threshold = float(joblib.load(ART_THRESHOLD))
        feature_cols = list(joblib.load(ART_FEATURES))
        meta = {"trained": False}
        return model, threshold, feature_cols, meta

    # Else, train quickly from Dataset.csv
    df = load_kaggle()
    if df.empty:
        # minimal fallback if no dataset
        X = pd.DataFrame({"HbA1c_level":[5.5, 8.0, 6.2, 7.4], "blood_glucose_level":[100, 240, 150, 200]})
        y = np.array([0,1,0,1])
        feature_cols = list(X.columns)
        pre = ColumnTransformer([("num","passthrough", feature_cols)])
        clf = GradientBoostingClassifier(random_state=42)
        model = Pipeline([("pre", pre), ("clf", CalibratedClassifierCV(clf, cv=3))]).fit(X, y)
        threshold = 0.5
        meta = {"trained": True, "auc": None}
        return model, threshold, feature_cols, meta

    # Clean target type
    if "diabetes" not in df.columns:
        st.error("Target column 'diabetes' not found in Dataset.csv")
        st.stop()
    df = df.copy()
    df["diabetes"] = pd.to_numeric(df["diabetes"], errors="coerce").fillna(0).astype(int)

    # Basic feature selection: keep useful clinical columns if present
    preferred_cols = [
        "age","hypertension","heart_disease","bmi",
        "HbA1c_level","blood_glucose_level",
        "gender","smoking_history"
    ]
    kept = [c for c in preferred_cols if c in df.columns]
    if not kept:
        kept = [c for c in df.columns if c != "diabetes"]

    X = df[kept].copy()
    y = df["diabetes"].values

    pre, nums, cats = build_preprocessor(pd.concat([X, df["diabetes"]], axis=1))
    clf = GradientBoostingClassifier(random_state=42)
    model = Pipeline([("pre", pre), ("clf", CalibratedClassifierCV(clf, cv=3))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(Xtr, ytr)

    yprob = model.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, yprob)
    thr = pick_threshold(yte, yprob)

    feature_cols = kept

    # persist artifacts for reproducibility
    try:
        joblib.dump(model, ART_MODEL)
        joblib.dump(thr, ART_THRESHOLD)
        joblib.dump(feature_cols, ART_FEATURES)
    except Exception:
        pass

    meta = {"trained": True, "auc": float(auc)}
    return model, float(thr), feature_cols, meta

def predict_row(model: Pipeline, feature_cols: List[str], row: pd.Series) -> float:
    x = pd.DataFrame([row.get(c, np.nan) for c in feature_cols], index=feature_cols).T
    try:
        prob = float(model.predict_proba(x)[:, 1][0])
    except Exception:
        prob = 0.0
    return prob

# --------- KPIs & Advice (FIXED TIR + clearer messages) ---------

def compute_kpis(g: pd.DataFrame, threshold: float):
    """
    Returns: tir(%), alerts(count), mean_glucose, max_prob
    TIR is 70–180 mg/dL (inclusive).
    """
    if g is None or g.empty or "blood_glucose_level" not in g:
        return None, 0, np.nan, np.nan
    gl = g["blood_glucose_level"]
    tir = float(((gl >= 70) & (gl <= 180)).mean() * 100.0)
    alerts = int((g.get("proba", pd.Series(dtype=float)) >= threshold).sum()) if "proba" in g else 0
    mean_glucose = float(gl.mean())
    max_prob = float(g.get("proba", pd.Series([0.0])).max()) if "proba" in g else np.nan
    return tir, alerts, mean_glucose, max_prob

def advise(glucose: float, hba1c: float) -> str:
    """Simple, conservative, non-medical advice string."""
    if glucose < 70:
        return "🟠 Low: Treat hypoglycaemia per your care plan (15–20g fast carbs, recheck in 15 min)."
    if glucose < 140 and hba1c < 6:
        return "🟢 Normal: Maintain healthy diet and regular exercise."
    if 140 <= glucose < 180 or (6 <= hba1c < 7):
        return "⚠️ Slightly elevated: Recheck in a few hours and limit sugar intake."
    if glucose >= 180 or hba1c >= 7:
        return "🔴 High: Monitor closely and consult your clinician if persistent."
    return "ℹ️ Please verify readings."

# --------- Forecasting (Hold-Last, EWMA, ARIMA) ---------

def forecast_hold_last(series: pd.Series, steps: int = 6) -> List[float]:
    if series.empty:
        return []
    last = float(series.iloc[-1])
    return [last] * steps

def forecast_ewma(series: pd.Series, steps: int = 6, alpha: float = 0.3) -> List[float]:
    if series.empty:
        return []
    # EWMA level
    level = float(series.iloc[0])
    for v in series.iloc[1:]:
        level = alpha * float(v) + (1 - alpha) * level
    return [level] + [level] * (steps - 1)

def forecast_arima(series: pd.Series, steps: int = 6) -> List[float]:
    if not _HAS_ARIMA or len(series) < 8:
        return forecast_ewma(series, steps=steps, alpha=0.4)
    try:
        # Simple (1,1,1) or auto? Keep static for stability on Streamlit Cloud
        model = ARIMA(series.astype(float), order=(1, 1, 1))
        fit = model.fit()
        fc = fit.forecast(steps=steps)
        return [float(x) for x in fc.values]
    except Exception:
        return forecast_ewma(series, steps=steps, alpha=0.4)

# --------- UI State ---------

def init_state():
    if "log_df" not in st.session_state:
        st.session_state.log_df = pd.DataFrame(columns=[
            "patient_id","historical_ts","shifted_ts","arrival_ts","shown_ts",
            "blood_glucose_level","HbA1c_level","proba","pred","status","advice"
        ])
    if "ptr" not in st.session_state:
        st.session_state.ptr = {}  # per patient pointer for streaming

# --------- Pages ---------

def page_overview(collected: pd.DataFrame, model, threshold: float):
    st.subheader("🗺️ Command Center")

    if collected.empty:
        st.info("No collected dataset found. Place `Collected_cleaned.csv` or `Collected_dataset.csv` in the app folder.")
        return

    pats = sorted(collected["patient_id"].dropna().unique().tolist())
    sel = st.multiselect("Select patients", pats, default=pats[: min(5, len(pats))])

    if not sel:
        st.warning("Please pick at least one patient.")
        return

    # KPIs across selection
    g = st.session_state.log_df
    if not g.empty:
        gsel = g[g["patient_id"].isin(sel)]
    else:
        gsel = collected[collected["patient_id"].isin(sel)].copy()
    tir, alerts, mean_glucose, max_prob = compute_kpis(gsel, threshold)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time in Range (70–180)", f"{tir:.1f}%" if tir is not None else "—")
    c2.metric("Alerts (≥thr)", f"{alerts}")
    c3.metric("Mean Glucose", f"{mean_glucose:.0f} mg/dL" if not np.isnan(mean_glucose) else "—")
    c4.metric("Max Risk Prob", f"{max_prob:.2f}" if not np.isnan(max_prob) else "—")

    # Simple plot
    plot_df = collected[collected["patient_id"].isin(sel)].copy()
    st.line_chart(
        plot_df.sort_values("shifted_ts").set_index("shifted_ts")[["blood_glucose_level"]],
        height=260
    )

    st.caption("This is a **simulated real-time replay** aligned to Europe/London. Forecasts are future projections. Not a medical device.")

def page_patient_twin(collected: pd.DataFrame, model, threshold: float, feature_cols: List[str]):
    st.subheader("🧍 Patient Twin")

    if collected.empty:
        st.info("No collected dataset found.")
        return

    pats = sorted(collected["patient_id"].dropna().unique().tolist())
    pid = st.selectbox("Patient", pats)

    dfp = collected[collected["patient_id"] == pid].sort_values("shifted_ts").reset_index(drop=True)
    if pid not in st.session_state.ptr:
        st.session_state.ptr[pid] = 0

    # Sidebar controls
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

        st.markdown("### ➕ Add live reading")
        with st.form("add_reading"):
            new_gl = st.number_input("Glucose (mg/dL)", min_value=0, max_value=600, value=120, step=1)
            new_hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=15.0, value=6.0, step=0.1, format="%.1f")
            submitted = st.form_submit_button("Append & Tick")
            if submitted:
                new = {
                    "patient_id": pid,
                    "timestamp": pd.Timestamp(now_uk()),
                    "shifted_ts": pd.Timestamp(now_uk()),
                    "blood_glucose_level": float(new_gl),
                    "HbA1c_level": float(new_hba1c),
                }
                proba = predict_row(model, feature_cols, pd.Series(new))
                pred = int(proba >= threshold)
                status = "🔴 DIABETIC" if pred else "🟢 NORMAL"

                rec = {
                    "patient_id": new["patient_id"],
                    "historical_ts": new["timestamp"],
                    "shifted_ts": new["shifted_ts"],
                    "arrival_ts": pd.Timestamp(now_uk()),
                    "shown_ts": new["shifted_ts"],
                    "blood_glucose_level": new["blood_glucose_level"],
                    "HbA1c_level": new["HbA1c_level"],
                    "proba": proba,
                    "pred": pred,
                    "status": status,
                    "advice": advise(new["blood_glucose_level"], new["HbA1c_level"]),
                }
                st.session_state.log_df = pd.concat([st.session_state.log_df, pd.DataFrame([rec])], ignore_index=True)
                try:
                    append_to_csv("live_predictions_log.csv", pd.DataFrame([rec]))
                except Exception:
                    pass
                st.success("Added one live reading.")

    # Current log for this patient
    lg = st.session_state.log_df[st.session_state.log_df["patient_id"] == pid].copy()

    # KPIs
    tir, alerts, mean_glucose, max_prob = compute_kpis(lg, threshold)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TIR (70–180)", f"{tir:.1f}%" if tir is not None else "—")
    c2.metric("Alerts (≥thr)", f"{alerts}")
    c3.metric("Mean Glucose", f"{mean_glucose:.0f} mg/dL" if not np.isnan(mean_glucose) else "—")
    c4.metric("Max Risk Prob", f"{max_prob:.2f}" if not np.isnan(max_prob) else "—")

    # Plot timeline
    if not lg.empty:
        disp = lg.sort_values("shown_ts").set_index("shown_ts")
        st.line_chart(disp[["blood_glucose_level"]], height=260)
        st.caption("Legend: 🟢 NORMAL, 🔴 DIABETIC")

    # What-if forecast
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

    st.info("This app is an educational prototype. It provides advisory text only — **not a medical device**.")

def add_next_tick(pid: str, dfp: pd.DataFrame, model, threshold: float, feature_cols: List[str]):
    ptr = st.session_state.ptr.get(pid, 0)
    if ptr >= len(dfp):
        st.warning("No more historical ticks for this patient.")
        return
    row = dfp.iloc[ptr]
    st.session_state.ptr[pid] = ptr + 1

    # Build features for classification prob
    feat = pd.Series({
        "patient_id": row.get("patient_id"),
        "blood_glucose_level": float(row.get("blood_glucose_level", np.nan)),
        "HbA1c_level": float(row.get("HbA1c_level", np.nan)),
        # Add other columns if your model expects them (e.g., age, bmi) — left as NaN if not available
    })

    proba = predict_row(model, feature_cols, feat)
    pred = int(proba >= threshold)
    status = "🔴 DIABETIC" if pred else "🟢 NORMAL"

    rec = {
        "patient_id": row.get("patient_id"),
        "historical_ts": row.get("timestamp", row.get("shifted_ts")),
        "shifted_ts": row.get("shifted_ts"),
        "arrival_ts": pd.Timestamp(now_uk()),
        "shown_ts": row.get("shifted_ts"),
        "blood_glucose_level": float(row.get("blood_glucose_level", np.nan)),
        "HbA1c_level": float(row.get("HbA1c_level", np.nan)),
        "proba": proba,
        "pred": pred,
        "status": status,
        "advice": advise(float(row.get("blood_glucose_level", np.nan)), float(row.get("HbA1c_level", np.nan))),
    }

    st.session_state.log_df = pd.concat([st.session_state.log_df, pd.DataFrame([rec])], ignore_index=True)
    try:
        append_to_csv("live_predictions_log.csv", pd.DataFrame([rec]))
    except Exception:
        pass

# --------- Model page ---------

def page_model_info(model_meta: Dict, threshold: float):
    st.subheader("🧠 Model & Threshold")
    st.write("Classifier: Gradient Boosting (calibrated) or loaded artifact.")
    if model_meta.get("auc") is not None:
        st.metric("Validation AUC", f"{model_meta['auc']:.3f}")
    else:
        st.caption("AUC not available (loaded artifacts or tiny fallback training).")
    st.metric("Operating Threshold", f"{threshold:.3f}")
    st.caption("Threshold chosen by maximizing F1 on validation PR curve, then persisted as an artifact.")

# --------- Logs page ---------

def page_logs():
    st.subheader("📜 Data & Logs")
    lg = st.session_state.log_df.copy()
    if lg.empty:
        st.info("Log is empty. Use Patient Twin → Next tick or Add live reading.")
        return
    st.dataframe(lg.sort_values("shown_ts"), use_container_width=True)
    # Download
    csv = lg.to_csv(index=False).encode("utf-8")
    st.download_button("Download log CSV", data=csv, file_name="live_predictions_log.csv", mime="text/csv")

# --------- Main ---------

def main():
    st.set_page_config(page_title="Digital Twin for Real-Time Diabetes", page_icon="🩸", layout="wide")
    st.title("Digital Twin Framework — Real-Time Diabetes (Simulated)")

    st.markdown(
        "> This demo **replays collected data on a real-time clock** and supports "
        "**short-term forecasting** (Hold-Last, EWMA, ARIMA). "
        "Swap the CSV stream with a device/API to make it fully live."
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

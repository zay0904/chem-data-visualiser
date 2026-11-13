"""
Streamlit web app for the Chemistry Data Visualizer.

Run locally:
    streamlit run app.py

Deploy free: https://streamlit.io/cloud
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analyzer import (
    load_csv,
    fit_linear,
    calculate_rates,
    beer_lambert,
    half_life_first_order,
)


st.set_page_config(page_title="Chemistry Data Visualizer",
                   page_icon="🧪", layout="wide")

st.title("🧪 Chemistry Data Visualizer")
st.caption("Load experimental data, fit trends, extract rate constants, and export report-ready plots.")

# ---------------- Sidebar: data source ----------------
st.sidebar.header("1. Data")
source = st.sidebar.radio("Source", ["Sample data", "Upload CSV/Excel", "Paste CSV"])

df = None
if source == "Sample data":
    df = load_csv("data/sample_kinetics.csv")
    st.sidebar.success("Loaded sample first-order kinetics data")
elif source == "Upload CSV/Excel":
    up = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])
    if up:
        try:
            df = load_csv(io.BytesIO(up.read()) if up.name.endswith(".csv") else up)
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")
else:
    text = st.sidebar.text_area("Paste CSV (with header row)", height=200)
    if text.strip():
        try:
            df = load_csv(io.StringIO(text))
        except Exception as e:
            st.sidebar.error(f"Could not parse CSV: {e}")

if df is None:
    st.info("Choose a data source in the sidebar to begin.")
    st.stop()

st.subheader("Preview")
st.dataframe(df, use_container_width=True, height=220)

# ---------------- Column selection ----------------
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if len(numeric_cols) < 2:
    st.error("Need at least two numeric columns.")
    st.stop()

st.sidebar.header("2. Plot")
x_col = st.sidebar.selectbox("X column", numeric_cols, index=0)
y_col = st.sidebar.selectbox("Y column", numeric_cols,
                             index=1 if len(numeric_cols) > 1 else 0)
x_label = st.sidebar.text_input("X label", x_col)
y_label = st.sidebar.text_input("Y label", y_col)
title = st.sidebar.text_input("Title", f"{y_col} vs {x_col}")
show_fit = st.sidebar.checkbox("Show linear fit", value=True)
log_y = st.sidebar.checkbox("Plot ln(y) — for first-order kinetics", value=False)

x = df[x_col].to_numpy(dtype=float)
y = df[y_col].to_numpy(dtype=float)

if log_y and (y <= 0).any():
    st.error("ln(y) requires all y values to be positive.")
    st.stop()

y_plot = np.log(y) if log_y else y

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(x, y_plot, s=45, color="#2575fc", edgecolor="#1c65e0",
           zorder=3, label="Data")

fit = None
if show_fit and len(x) >= 2:
    try:
        fit = fit_linear(x, y_plot)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, fit.slope * x_line + fit.intercept,
                color="#e63946", lw=2,
                label=f"{fit.equation}   (R² = {fit.r_squared:.4f})")
    except Exception as e:
        st.warning(f"Fit failed: {e}")

ax.set_xlabel(x_label)
ax.set_ylabel(f"ln({y_label})" if log_y else y_label)
ax.set_title(title)
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()

st.subheader("Plot")
st.pyplot(fig)

# ---------------- Metrics ----------------
st.subheader("Results")
c1, c2, c3, c4 = st.columns(4)
if fit:
    c1.metric("Slope", f"{fit.slope:.5g}")
    c2.metric("Intercept", f"{fit.intercept:.5g}")
    c3.metric("R²", f"{fit.r_squared:.5f}")
    c4.metric("n", fit.n)

if log_y and fit and fit.slope < 0:
    k = -fit.slope
    st.success(
        f"**First-order kinetics detected.**  "
        f"Rate constant k ≈ {k:.4g} s⁻¹  •  "
        f"Half-life t₁/₂ ≈ {half_life_first_order(k, 's')}"
    )

# ---------------- Reaction rates ----------------
with st.expander("Reaction rate (dC/dt) from concentration data"):
    st.write("Computes -dC/dt using central differences. "
             "Requires a concentration column and a time column.")
    conc_col = st.selectbox("Concentration column", numeric_cols, key="conc")
    time_col = st.selectbox("Time column", numeric_cols,
                             index=min(1, len(numeric_cols) - 1), key="time")
    if st.button("Compute rates"):
        try:
            rates = calculate_rates(df[conc_col], df[time_col])
            rate_df = pd.DataFrame({
                "time": df[time_col],
                "concentration": df[conc_col],
                "rate": rates,
            })
            st.dataframe(rate_df, use_container_width=True)
            st.download_button(
                "Download rates CSV",
                rate_df.to_csv(index=False).encode(),
                "rates.csv", "text/csv",
            )
        except Exception as e:
            st.error(str(e))

# ---------------- Beer–Lambert ----------------
with st.expander("Beer–Lambert concentration calculator"):
    st.write("A = ε · c · l  →  c = A / (ε · l)")
    col_a, col_b, col_c = st.columns(3)
    A = col_a.number_input("Absorbance (A)", min_value=0.0, value=0.450, step=0.01)
    eps = col_b.number_input("Molar absorptivity ε (M⁻¹·cm⁻¹)",
                              min_value=0.1, value=15000.0, step=100.0)
    l_cm = col_c.number_input("Path length l (cm)", min_value=0.1,
                               value=1.0, step=0.1)
    if st.button("Calculate concentration"):
        try:
            conc = beer_lambert(A, eps, l_cm)
            st.success(f"c = {conc:.6g} mol/L  ({conc * 1e3:.4g} mmol/L)")
        except Exception as e:
            st.error(str(e))

# ---------------- Downloads ----------------
st.subheader("Downloads")
png_buf = io.BytesIO()
fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight")
png_buf.seek(0)

csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode()

d1, d2 = st.columns(2)
d1.download_button("⬇ Download plot (PNG)", png_buf.getvalue(),
                    "plot.png", "image/png")
d2.download_button("⬇ Download data (CSV)", csv_bytes,
                    "data.csv", "text/csv")

st.caption("Built by Mwila Molobeka — chemistry meets code.")
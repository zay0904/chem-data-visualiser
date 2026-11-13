# 🧪 Chemistry Data Visualizer

A small, focused tool for chemistry students and lab researchers to
analyze experimental data, fit trends, extract rate constants, and
export report-ready plots. Written in Python (with a live Streamlit
web version), it turns raw CSV/Excel instrument exports into a clean,
labeled figure with the fitted equation and R².

![Demo screenshot](screenshots/demo.png)

## Features

- **Load** CSV, Excel, or pasted data with automatic type detection.
- **Plot** X vs Y with scatter + linear fit, equation and R² overlay.
- **First-order kinetics**: plot ln[A] vs time to extract rate constant `k`
  and half-life `t₁/₂`.
- **Reaction rate** (`-dC/dt`) via central differences.
- **Beer–Lambert** concentration calculator.
- **Export** processed data as CSV and the plot as PNG.
- **Tests** with pytest (8 unit tests).

## Quick start

```bash
pip install -r requirements.txt

# CLI: fit absorbance vs time
python cli.py --input data/sample_kinetics.csv \
              --x time_s --y absorbance \
              --out results/absorbance.png

# CLI: first-order kinetics, ln[A] vs time
python cli.py --input data/sample_kinetics.csv \
              --x time_s --y concentration_M --logy \
              --out results/kinetics.png

# Web app (Streamlit)
streamlit run app.py

# Tests
pytest -q# chem-data-visuals

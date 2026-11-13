"""
Core analysis functions for the Chemistry Data Visualizer.

Pure functions only — no printing, no plotting side effects beyond
returning objects. This makes them reusable from the CLI, the
Streamlit app, and the tests.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- loading
def load_csv(path_or_buffer) -> pd.DataFrame:
    """
    Load a CSV (path or file-like) and return a cleaned DataFrame.

    - Accepts .csv and .xlsx (pass .xlsx path or a buffer for CSV text).
    - Coerces numeric columns to float, drops fully-empty rows.
    - Raises ValueError if no numeric columns are found.
    """
    p = Path(path_or_buffer) if isinstance(path_or_buffer, (str, Path)) else None

    if p and p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(path_or_buffer)

    df = df.dropna(how="all").reset_index(drop=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    if not any(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
        raise ValueError("No numeric columns found in the file.")

    return df


# ---------------------------------------------------------------- rates
def calculate_rates(concentration, time) -> np.ndarray:
    """
    Reaction rate via numerical differentiation: -dC/dt.

    Uses central differences (np.gradient), which is more accurate than
    simple forward/backward differences. The result has the same shape
    as the input.
    """
    c = np.asarray(concentration, dtype=float)
    t = np.asarray(time, dtype=float)

    if c.shape != t.shape:
        raise ValueError("concentration and time must have the same length")
    if len(t) < 2:
        raise ValueError("Need at least 2 data points to compute a rate")
    if not np.all(np.diff(t) > 0):
        raise ValueError("time must be strictly increasing")

    return -np.gradient(c, t)


# ---------------------------------------------------------------- fitting
@dataclass
class LinearFit:
    """Result of a linear regression."""
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    stderr: float
    n: int

    @property
    def equation(self) -> str:
        sign = "+" if self.intercept >= 0 else "−"
        return f"y = {self.slope:.4g}x {sign} {abs(self.intercept):.4g}"


def fit_linear(x, y) -> LinearFit:
    """
    Ordinary least-squares linear fit y = m·x + b.

    Returns slope, intercept, R², p-value, standard error of the slope,
    and n. Uses scipy.stats.linregress.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        raise ValueError("Need at least 2 points to fit a line")

    res = stats.linregress(x, y)
    return LinearFit(
        slope=res.slope,
        intercept=res.intercept,
        r_squared=res.rvalue ** 2,
        p_value=res.pvalue,
        stderr=res.stderr,
        n=len(x),
    )


# ---------------------------------------------------------------- chemistry
def beer_lambert(absorbance, molar_absorptivity, path_length_cm=1.0) -> float:
    """
    Beer–Lambert law: A = ε · c · l  →  c = A / (ε · l)

    Parameters
    ----------
    absorbance : float
    molar_absorptivity : float   # ε, in M⁻¹·cm⁻¹
    path_length_cm : float       # l, in cm (default 1.0)

    Returns
    -------
    Concentration in mol/L (M).
    """
    if molar_absorptivity <= 0 or path_length_cm <= 0:
        raise ValueError("ε and path length must be positive")
    if absorbance < 0:
        raise ValueError("Absorbance cannot be negative")
    return absorbance / (molar_absorptivity * path_length_cm)


def half_life_first_order(k: float, unit: str = "s") -> str:
    """t₁/₂ = ln(2) / k, returned as a human-readable string."""
    if k <= 0:
        raise ValueError("Rate constant k must be positive")
    t_half = np.log(2) / k
    return f"{t_half:.4g} {unit}"


# ---------------------------------------------------------------- plotting
def plot_results(
    x,
    y,
    x_label: str = "Time",
    y_label: str = "Concentration",
    title: str = "Chemistry Data",
    fit: bool = True,
    log_y: bool = False,
    save_path: str | Path | None = None,
):
    """
    Plot data (scatter) with an optional linear fit and equation overlay.

    Returns the matplotlib Figure. If save_path is given, also saves a PNG.
    Uses a non-interactive backend so it works in scripts and on servers.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    if log_y:
        y = np.log(y)
        y_label = f"ln({y_label})"

    ax.scatter(x, y, s=40, color="#2575fc", edgecolor="#1c65e0",
               zorder=3, label="Data")

    if fit:
        f = fit_linear(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = f.slope * x_line + f.intercept
        ax.plot(x_line, y_line, color="#e63946", linewidth=2,
                label=f"{f.equation}   (R² = {f.r_squared:.4f})")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------- export
def export_results(
    df: pd.DataFrame,
    fit: LinearFit | None = None,
    extra: dict | None = None,
    outdir: str | Path = "results",
) -> dict[str, Path]:
    """
    Write a summary CSV and (if a fit is given) a fit-summary CSV.

    Returns a dict of the files written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    data_path = outdir / "processed_data.csv"
    df.to_csv(data_path, index=False)
    written["data"] = data_path

    if fit is not None:
        summary = pd.DataFrame([{
            "slope": fit.slope,
            "intercept": fit.intercept,
            "r_squared": fit.r_squared,
            "p_value": fit.p_value,
            "stderr_slope": fit.stderr,
            "n": fit.n,
        }])
        if extra:
            for k, v in extra.items():
                summary[k] = v
        fit_path = outdir / "fit_summary.csv"
        summary.to_csv(fit_path, index=False)
        written["fit"] = fit_path

    return written
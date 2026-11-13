#!/usr/bin/env python3
"""
Command-line interface for the Chemistry Data Visualizer.

Examples
--------
Plot absorbance vs. time with a linear fit:

    python cli.py --input data/sample_kinetics.csv \\
                  --x time_s --y absorbance \\
                  --xlabel "Time (s)" --ylabel "Absorbance" \\
                  --out results/absorbance.png

First-order kinetics: fit ln[A] vs. time to extract k:

    python cli.py --input data/sample_kinetics.csv \\
                  --x time_s --y concentration_M --logy \\
                  --xlabel "Time (s)" --ylabel "[A] (M)" \\
                  --out results/kinetics.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.analyzer import (
    load_csv,
    fit_linear,
    plot_results,
    export_results,
    half_life_first_order,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Chemistry Data Visualizer (CLI)")
    ap.add_argument("--input", required=True, help="CSV or XLSX file")
    ap.add_argument("--x", required=True, help="X column name")
    ap.add_argument("--y", required=True, help="Y column name")
    ap.add_argument("--xlabel", default=None)
    ap.add_argument("--ylabel", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--logy", action="store_true",
                    help="Plot ln(y) — useful for first-order kinetics")
    ap.add_argument("--out", default="results/plot.png",
                    help="Output PNG path")
    ap.add_argument("--no-fit", action="store_true",
                    help="Don't overlay a linear fit")
    args = ap.parse_args()

    df = load_csv(args.input)
    for col in (args.x, args.y):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found. "
                             f"Available: {list(df.columns)}")

    x = df[args.x].to_numpy()
    y = df[args.y].to_numpy()

    # For log-y, we need positive y values
    if args.logy and (y <= 0).any():
        raise SystemExit("--logy requires all y values to be positive.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_results(
        x, y,
        x_label=args.xlabel or args.x,
        y_label=args.ylabel or args.y,
        title=args.title or f"{args.y} vs {args.x}",
        fit=not args.no_fit,
        log_y=args.logy,
        save_path=out_path,
    )

    # Print a text summary
    y_for_fit = __import__("numpy").log(y) if args.logy else y
    f = fit_linear(x, y_for_fit)
    print(f"\n✓ Saved plot: {out_path}")
    print(f"  Fit: {f.equation}")
    print(f"  R²:  {f.r_squared:.5f}   (n = {f.n})")
    print(f"  Slope stderr: {f.stderr:.4g}")

    # If we just fit ln[A] vs t, report k and half-life
    if args.logy:
        k = -f.slope
        if k > 0:
            print(f"  First-order k ≈ {k:.4g} s⁻¹")
            print(f"  Half-life t₁/₂ ≈ {half_life_first_order(k, 's')}")

    files = export_results(df, fit=f, outdir=out_path.parent)
    for name, path in files.items():
        print(f"  Wrote {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
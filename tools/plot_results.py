#!/usr/bin/env python3
"""Render the error-vs-hit-rate plot from the Go benchmark's results.json.

Presentation only — the benchmark LOGIC is entirely in Go (bench/); this script
just draws what the harness computed, same justification as the prep script. If
you'd rather avoid Python entirely, gonum/plot is the pure-Go alternative
(github.com/gonum/plot) — it would live in the Go harness and emit the PNG
directly.

x-axis = error rate (FP/n), y-axis = hit rate ((TP+FP)/n). Static-threshold and
verified arms are drawn together; each delta's error budget is a vertical dashed
line. A verified point must sit at or left of its own delta line — that is the
guarantee, visualized.

Usage:
    .venv/bin/python tools/plot_results.py --results bench/results.json --out bench/error_vs_hit.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def ci_bars(points, rate_key, lo_key, hi_key):
    """Return [lower_err, upper_err] arrays for matplotlib error bars."""
    lo = [max(0.0, p[rate_key] - p[lo_key]) for p in points]
    hi = [max(0.0, p[hi_key] - p[rate_key]) for p in points]
    return [lo, hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench/results.json")
    ap.add_argument("--out", default="bench/error_vs_hit.png")
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    runs = data["runs"]
    static = sorted([r for r in runs if r["arm"] == "static"], key=lambda r: r["error_rate"])
    verified = sorted([r for r in runs if r["arm"] == "verified"], key=lambda r: r["param"])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Static arm: connected curve with Wilson error bars.
    if static:
        ax.errorbar(
            [r["error_rate"] for r in static],
            [r["hit_rate"] for r in static],
            xerr=ci_bars(static, "error_rate", "error_ci_lo", "error_ci_hi"),
            yerr=ci_bars(static, "hit_rate", "hit_ci_lo", "hit_ci_hi"),
            fmt="o-", color="#c44", ecolor="#c44", elinewidth=0.8, capsize=2,
            label="static threshold (GPTCache/LiteLLM baseline)", zorder=3,
        )
        for r in static:
            ax.annotate(f"T={r['param']:.2f}", (r["error_rate"], r["hit_rate"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7, color="#833")

    # Verified arm: points with error bars.
    if verified:
        ax.errorbar(
            [r["error_rate"] for r in verified],
            [r["hit_rate"] for r in verified],
            xerr=ci_bars(verified, "error_rate", "error_ci_lo", "error_ci_hi"),
            yerr=ci_bars(verified, "hit_rate", "hit_ci_lo", "hit_ci_hi"),
            fmt="s", color="#26a", ecolor="#26a", elinewidth=0.8, capsize=2,
            markersize=8, label="verified policy (semblance)", zorder=4,
        )
        for r in verified:
            ax.annotate(f"δ={r['param']:.2f}", (r["error_rate"], r["hit_rate"]),
                        textcoords="offset points", xytext=(6, -2), fontsize=8, color="#148")

    # Delta budget lines: each verified run must sit at or left of its line.
    for r in verified:
        d = r["param"]
        ax.axvline(d, color="#26a", linestyle="--", linewidth=0.9, alpha=0.5)
        ax.annotate(f"budget δ={d:.2f}", (d, ax.get_ylim()[1]),
                    rotation=90, va="top", ha="right", fontsize=7, color="#26a", alpha=0.8)

    ax.set_xlabel("error rate  (FP / n)")
    ax.set_ylabel("cache hit rate  ((TP + FP) / n)")
    n = data.get("n")
    cap = data.get("capacity", 0)
    cap_s = "unbounded" if not cap else str(cap)
    dataset = data.get("dataset", "")
    label = "LmArena" if "lmarena" in dataset.lower() else ("SearchQueries" if "search" in dataset.lower() else dataset)
    nmin = data.get("nmin")
    ax.set_title(
        f"Verified semantic caching vs static threshold\n"
        f"{label} replay — n={n}, capacity={cap_s}, seed={data.get('seed')}, n_min={nmin}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.margins(x=0.08, y=0.08)
    if ax.get_xlim()[0] > 0:
        ax.set_xlim(left=0)
    if ax.get_ylim()[0] > 0:
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

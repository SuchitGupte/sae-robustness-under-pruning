"""
4 + 2 Experiments comparing Magnitude, SparseGPT, and Wanda pruning on SAE quality.

Exp 1  — Core Reconstruction:    % change from baseline vs layer (sparsity=50)
Exp 2  — Interpretability:       % change from baseline vs layer (sparsity=50)
Exp 3  — Sparsity Scaling:       mean % change vs sparsity level (25/50/75)
Exp 4  — Aggregate Ranking:      mean % change per metric, all methods (sparsity=50)
Exp 5  — Per-Layer × Sparsity:   for each metric, one figure; subplots = layers;
                                   x = sparsity level, y = raw metric value
Exp 6  — Layer-wise Degradation: raw metric values across layers at sparsity=50
                                   (dense + 3 methods) to reveal layer trends
Exp 12 — SCR & TPP Threshold Comparison:
           For thresholds [2,5,10,20,50,100,500] at sparsity=50:
             (a) mean raw value vs threshold  (dense + 3 methods)
             (b) mean % change from dense vs threshold  (3 methods)
           Four subplots: SCR raw / SCR % change / TPP raw / TPP % change
"""

import json
import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── Paper style ───────────────────────────────────────────────────────────────
# plt.rcParams.update({
#     "font.family":               "Liberation Serif",
#     "font.size":                 10,
#     "axes.titlesize":            11,
#     "axes.titleweight":          "bold",
#     "axes.labelsize":            10,
#     "xtick.labelsize":           9,
#     "ytick.labelsize":           9,
#     "legend.fontsize":           9,
#     "legend.frameon":            False,
#     "legend.handlelength":       1.8,
#     "legend.handleheight":       0.8,
#     "axes.spines.top":           False,
#     "axes.spines.right":         False,
#     "axes.linewidth":            0.8,
#     "axes.grid":                 True,
#     "grid.linestyle":            ":",
#     "grid.alpha":                0.35,
#     "grid.linewidth":            0.6,
#     "lines.linewidth":           1.8,
#     "lines.markersize":          5,
#     "figure.dpi":                150,
#     "savefig.dpi":               200,
#     "savefig.bbox":              "tight",
# })

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "gemma_2_2b"
BASE_ROOT  = f"/research/nfs_khalili_17/gupte.31/reproduce/SAEBench/{MODEL_NAME}/canonical"
OUT_DIR    = f"/research/nfs_khalili_17/gupte.31/reproduce/results/{MODEL_NAME}/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# METHODS    = ["magnitude", "sparsegpt", "wanda"]
METHODS    = [
    "magnitude",
    "sparsegpt", 
    "wanda"
    ]

# SPARSITIES = [25, 50, 75]
SPARSITIES = [25, 40, 50]
SCR_THRESHOLDS = [2, 5, 10, 20, 50, 100, 500]
TPP_THRESHOLDS = [2, 5, 10, 20, 50, 100, 500]
FONTDELTA = 7

METHOD_STYLES = {
    "magnitude": dict(color="#E63946", marker="o",  linestyle="-"),
    "sparsegpt": dict(color="#457B9D", marker="s",  linestyle="--"),
    "wanda":     dict(color="#2A9D8F", marker="^",  linestyle="-."),
}
DENSE_STYLE = dict(color="black", linestyle=":", linewidth=1.5)

# ── File-path helpers ─────────────────────────────────────────────────────────

def core_glob(sparsity, method):
    if sparsity == "dense":
        return os.path.join(
            BASE_ROOT, "dense/eval_results/core",
            "gemma-scope-2b-pt-res-canonical_layer_*_width_16k_canonical_eval_results.json",
        )
    return os.path.join(
        BASE_ROOT, f"sparsity={sparsity}/{method}/eval_results/core",
        "gemma-scope-2b-pt-res-canonical_layer_*_width_16k_canonical_eval_results.json",
    )

def interp_glob(sparsity, method, eval_type):
    sub = {"scr": "scr/scr", "tpp": "tpp/tpp", "absorption": "absorption"}[eval_type]
    if sparsity == "dense":
        return os.path.join(
            BASE_ROOT, f"dense/eval_results/{sub}",
            "gemma-scope-2b-pt-res-canonical_layer_*_width_16k_canonical_eval_results.json",
        )
    return os.path.join(
        BASE_ROOT, f"sparsity={sparsity}/{method}/eval_results/{sub}",
        "gemma-scope-2b-pt-res-canonical_layer_*_width_16k_canonical_eval_results.json",
    )

# ── Low-level loaders ─────────────────────────────────────────────────────────

def _layer_from_path(path):
    for part in os.path.basename(path).split("_"):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot extract layer: {path}")


def _load_value(path, *keys):
    with open(path) as f:
        d = json.load(f)
    node = d["eval_result_metrics"]
    for k in keys:
        node = node[k]
    return float(node)


def load_layers(pattern, *keys):
    """Returns {layer: value} for all matched files."""
    result = {}
    for p in glob.glob(pattern):
        result[_layer_from_path(p)] = _load_value(p, *keys)
    return result


def pct_change_series(baseline, pruned):
    """Absolute % change from baseline for common layers, sorted by layer."""
    common = sorted(set(baseline) & set(pruned))
    pct = [abs(pruned[l] - baseline[l]) / abs(baseline[l]) * 100 for l in common]
    return common, pct


def mean_pct_change(baseline, pruned):
    layers, pct = pct_change_series(baseline, pruned)
    return float(np.mean(pct)) if pct else float("nan")

# ── Metric specs ──────────────────────────────────────────────────────────────

CORE_METRICS = [
    ("KL Div Score",       "model_behavior_preservation",    "kl_div_score"),
    ("CE Loss Score",      "model_performance_preservation", "ce_loss_score"),
    ("Explained Variance", "reconstruction_quality",         "explained_variance"),
    ("Cosine Similarity",  "reconstruction_quality",         "cossim"),
    # ("MSE",                "reconstruction_quality",         "mse"),
]

INTERP_METRICS = [
    ("Absorption Score", "absorption", "mean", "mean_full_absorption_score"),
    ("SCR (threshold=10)",        "scr",        "scr_metrics", "scr_metric_threshold_10"),
    ("TPP Total (threshold=10)",  "tpp",        "tpp_metrics", "tpp_threshold_10_total_metric"),
]

ALL_METRICS = CORE_METRICS + INTERP_METRICS

def get_glob_and_keys(metric_tuple):
    """Return (glob_fn, source_tag, *keys) from a metric tuple."""
    name = metric_tuple[0]
    if name in {t[0] for t in CORE_METRICS}:
        _, sec, key = metric_tuple
        return "core", sec, key
    else:
        _, eval_type, *keys = metric_tuple
        return ("interp", eval_type) + tuple(keys)

# ── Helper: build baseline dict ───────────────────────────────────────────────

def build_baseline(metric_tuple):
    name = metric_tuple[0]
    if name in {t[0] for t in CORE_METRICS}:
        _, sec, key = metric_tuple
        return load_layers(core_glob("dense", None), sec, key)
    else:
        _, eval_type, *keys = metric_tuple
        return load_layers(interp_glob("dense", None, eval_type), *keys)


def build_pruned(metric_tuple, sparsity, method):
    name = metric_tuple[0]
    if name in {t[0] for t in CORE_METRICS}:
        _, sec, key = metric_tuple
        return load_layers(core_glob(sparsity, method), sec, key)
    else:
        _, eval_type, *keys = metric_tuple
        return load_layers(interp_glob(sparsity, method, eval_type), *keys)

# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Core Reconstruction % Change vs Layer  (sparsity=50)
# ═══════════════════════════════════════════════════════════════════════════════
def exp1():
    print("=== Experiment 1: Core Reconstruction Metrics ===")

    fig, axes = plt.subplots(1, len(CORE_METRICS),figsize=(5 * len(CORE_METRICS), 5), constrained_layout=True)
    fig.suptitle("Core Reconstruction: % Change from Dense Baseline vs Layer  (sparsity=50%)", fontsize=FONTDELTA+20, fontweight="bold")

    for ax, metric in zip(axes, CORE_METRICS):
        name = metric[0]
        base = build_baseline(metric)
        for method in METHODS:
            pruned = build_pruned(metric, 50, method)
            layers, pct = pct_change_series(base, pruned)
            if layers:
                ax.plot(layers, _smooth(pct, 1.5), label=method, **METHOD_STYLES[method], linewidth=1.8, markersize=2)
        ax.set_xlabel("Layer", fontsize=FONTDELTA+10)
        ax.set_ylabel("% change from baseline", fontsize=FONTDELTA+10)
        ax.set_title(name, fontsize=FONTDELTA+12)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=FONTDELTA+10, framealpha=0.7)
        ax.tick_params(axis='both', labelsize=FONTDELTA+10)

    out1 = os.path.join(OUT_DIR, "exp1_core_delta_vs_layer.pdf")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out1}")

# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Interpretability % Change vs Layer  (sparsity=50)
# ═══════════════════════════════════════════════════════════════════════════════

def exp2():
    print("=== Experiment 2: Interpretability Metrics ===")

    fig, axes = plt.subplots(1, len(INTERP_METRICS),
                            figsize=(5.5 * len(INTERP_METRICS), 5),
                            constrained_layout=True)
    fig.suptitle("Interpretability: % Change from Dense Baseline vs Layer  (sparsity=50%)",
                fontsize=FONTDELTA+13, fontweight="bold")

    for ax, metric in zip(axes, INTERP_METRICS):
        name = metric[0]
        base = build_baseline(metric)
        for method in METHODS:
            pruned = build_pruned(metric, 50, method)
            layers, pct = pct_change_series(base, pruned)
            if layers:
                ax.plot(layers, _smooth(pct), label=method, **METHOD_STYLES[method],
                        linewidth=1.8, markersize=2)
        ax.set_xlabel("Layer", fontsize=FONTDELTA+11)
        ax.set_ylabel("% change from baseline", fontsize=FONTDELTA+11)
        ax.set_title(name, fontsize=FONTDELTA+12, fontweight="semibold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=FONTDELTA+9, framealpha=0.7)

    out2 = os.path.join(OUT_DIR, "exp2_interp_delta_vs_layer.pdf")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out2}")

def exp1_2_combined():
    print("=== Experiment 1+2: Core + Interpretability ===")

    mse_metric = ("MSE", "reconstruction_quality", "mse")
    core_no_mse = [m for m in CORE_METRICS if m[0] != "MSE"]
    interp_with_mse = list(INTERP_METRICS) + [mse_metric]
    all_metrics = [core_no_mse, interp_with_mse]

    n_rows = 2
    n_cols = max(len(core_no_mse), len(interp_with_mse))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        constrained_layout=True
    )

    fig.suptitle(
        "Core + Interpretability Metrics (% Change from Dense Baseline)",
        fontsize=FONTDELTA+20,
        fontweight="bold"
    )

    for r, metrics in enumerate(all_metrics):
        for c in range(n_cols):
            ax = axes[r, c]

            # If this row has fewer metrics, turn off extra axes
            if c >= len(metrics):
                ax.axis("off")
                continue

            metric = metrics[c]
            name = metric[0]
            base = build_baseline(metric)

            for method in METHODS:
                pruned = build_pruned(metric, 50, method)
                layers, pct = pct_change_series(base, pruned)
                if layers:
                    ax.plot(
                        layers,
                        _smooth(pct, 1.5 if r == 0 else 1.2),
                        label=method,
                        **METHOD_STYLES[method],
                        linewidth=1.8,
                        markersize=2
                    )

            ax.set_xlabel("Layer", fontsize=FONTDELTA+10)
            ax.set_ylabel("% change", fontsize=FONTDELTA+10)
            ax.set_title(name, fontsize=FONTDELTA+12, fontweight="normal")

            ax.grid(True, linestyle=":", alpha=0.5)
            ax.tick_params(axis='both', labelsize=FONTDELTA+10)

            ax.legend(fontsize=FONTDELTA+9, framealpha=0.7)

    out = os.path.join(OUT_DIR, "fig1.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 3.1 — Core Metrics: % Change vs Layer across Sparsities
# Experiment 3.2 — Interp Metrics: % Change vs Layer across Sparsities
#
# Layout: rows = sparsity levels (25 / 50 / 75),
#         cols = one subplot per metric.
# x-axis = layer, y-axis = % change from dense.  Lines = pruning methods.
# Mirrors the exp1 / exp2 style but adds the sparsity dimension as rows.
# ═══════════════════════════════════════════════════════════════════════════════

def _exp3_figure(metrics, title_prefix, out_name, sparsities=None):
    if sparsities is None:
        sparsities = SPARSITIES
    n_sp  = len(sparsities)
    n_met = len(metrics)
    fig, axes = plt.subplots(n_sp, n_met,
                             figsize=(5 * n_met, 4.5 * n_sp),
                             constrained_layout=True)
    fig.suptitle(f"{title_prefix}: % Change from Dense Baseline vs Layer",
                 fontsize=FONTDELTA+20, fontweight="bold")

    for ri, sp in enumerate(sparsities):
        for ci, metric in enumerate(metrics):
            ax   = axes[ri, ci]
            name = metric[0]
            base = build_baseline(metric)
            for method in METHODS:
                pruned = build_pruned(metric, sp, method)
                layers, pct = pct_change_series(base, pruned)
                if layers:
                    ax.plot(layers, _smooth(pct), label=method,
                            **METHOD_STYLES[method], linewidth=1.8, markersize=2)
            # Row label on leftmost subplot only
            if ci == 0:
                ax.set_ylabel(f"sparsity={sp}%\n% change from baseline",
                              fontsize=FONTDELTA+10)
            else:
                ax.set_ylabel("% change from baseline", fontsize=FONTDELTA+10)
            # Column title on top row only
            if ri == 0:
                ax.set_title(name, fontsize=FONTDELTA+12, fontweight="normal")
            ax.set_xlabel("Layer", fontsize=FONTDELTA+10)
            ax.tick_params(labelsize=FONTDELTA+10)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(fontsize=FONTDELTA+10, framealpha=0.7)

    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


EXP3_SPARSITIES = [10, 25, 40, 50, 60, 75]

def exp3_1():
    print("=== Experiment 3.1: Core Metrics — % Change vs Layer (sparsities 10/25/40/50/60/75) ===")
    _exp3_figure(CORE_METRICS, "Core Reconstruction",
                 "appfig1.pdf", EXP3_SPARSITIES)


def exp3_2():
    print("=== Experiment 3.2: Interpretability Metrics — % Change vs Layer (sparsities 10/25/40/50/60/75) ===")
    _exp3_figure(INTERP_METRICS, "Interpretability",
                 "appfig2.pdf", EXP3_SPARSITIES)

# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 4 — Layer-wise Degradation Analysis  (sparsity=50)
#
# One figure, 4×2 grid (7 metrics + 1 empty).
# x-axis = layer, y-axis = raw metric value.
# 4 lines: dense (black dashed), magnitude, sparsegpt, wanda.
# Shaded band = range between best and worst method at each layer.
# ═══════════════════════════════════════════════════════════════════════════════

def _smooth(y, sigma=1.2):
    """Gaussian smooth a 1-D array, ignoring NaNs."""
    from scipy.ndimage import gaussian_filter1d
    y = np.array(y, dtype=float)
    mask = np.isnan(y)
    if mask.all():
        return y
    # Fill NaN gaps with linear interpolation before smoothing
    xs = np.arange(len(y))
    y_filled = np.interp(xs, xs[~mask], y[~mask])
    y_smooth = gaussian_filter1d(y_filled, sigma=sigma)
    y_smooth[mask] = float("nan")
    return y_smooth


def exp4():
    print("=== Experiment 4: Layer-wise Degradation Analysis (sparsity=50) ===")

    NROWS4, NCOLS4 = 2, 4
    fig, axes = plt.subplots(NROWS4, NCOLS4, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("Layer-wise Degradation at Sparsity=50% (raw metric values: dense baseline + 3 pruning methods)", fontsize=FONTDELTA+20, fontweight="bold")

    axes_flat4 = axes.flatten()

    for idx, metric in enumerate(ALL_METRICS):
        ax = axes_flat4[idx]
        name = metric[0]

        base_vals = build_baseline(metric)
        layers    = sorted(base_vals.keys())

        # Dense baseline
        base_y = [base_vals[l] for l in layers]
        ax.plot(layers, _smooth(base_y, sigma=1.2),
                color="black", linestyle=":", linewidth=2.5, label="dense")

        method_vals = {}
        for method in METHODS:
            pruned = build_pruned(metric, 50, method)
            y = [pruned.get(l, float("nan")) for l in layers]
            method_vals[method] = y
            ax.plot(layers, _smooth(y), label=method, **METHOD_STYLES[method],
                    linewidth=2.2, markersize=2)

        # Shaded band: min-max across methods at each layer (smoothed)
        y_stack  = np.array(list(method_vals.values()), dtype=float)
        y_lo     = _smooth(np.nanmin(y_stack, axis=0))
        y_hi     = _smooth(np.nanmax(y_stack, axis=0))
        ax.fill_between(layers, y_lo, y_hi, alpha=0.12, color="gray",
                        label="method range")

        ax.set_xlabel("Layer", fontsize=FONTDELTA+10)
        ax.set_ylabel("Metric value", fontsize=FONTDELTA+10)
        ax.set_title(name, fontsize=FONTDELTA+12, fontweight="normal")
        ax.tick_params(labelsize=FONTDELTA+10)
        ax.grid(True, linestyle=":", alpha=0.4)

    # Single shared legend on the figure (avoids crowding each subplot)
    legend_handles = [
        mlines.Line2D([], [], color="black", linestyle=":", linewidth=2.5, label="dense"),
    ] + [
        mlines.Line2D([], [], color=METHOD_STYLES[m]["color"],
                      linestyle=METHOD_STYLES[m]["linestyle"],
                      marker=METHOD_STYLES[m]["marker"],
                      linewidth=2.2, markersize=6, label=m)
        for m in METHODS
    ] + [
        mlines.Line2D([], [], color="gray", linewidth=8, alpha=0.3, label="method range"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, fontsize=FONTDELTA+10, framealpha=0.85,
               bbox_to_anchor=(0.5, -0.04))

    out4 = os.path.join(OUT_DIR, "fig2.pdf")
    fig.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out4}")

# ═══════════════════════════════════════════════════════════════════════════════
# PRE-LOAD: build a single cache of all (metric, method, sparsity) → {layer: pct}
#           and (metric, method, sparsity) → {layer: raw_value}
#           so the new experiments below don't re-read files repeatedly.
# ═══════════════════════════════════════════════════════════════════════════════
def helpV():
    print("\n=== Pre-loading data cache for additional experiments ===")
    global pct_cache, ALL_LAYERS, raw_cache, LAYER_GROUPS_3, LAYER_GROUPS_5, group_mean_pct

    # pct_cache[metric_name][method][sparsity]  = {layer: abs_pct_change}
    # raw_cache[metric_name]["dense"]           = {layer: value}
    # raw_cache[metric_name][method][sparsity]  = {layer: value}
    pct_cache = {m[0]: {meth: {} for meth in METHODS} for m in ALL_METRICS}
    raw_cache = {m[0]: {"dense": {}} for m in ALL_METRICS}
    for m in ALL_METRICS:
        raw_cache[m[0]].update({meth: {} for meth in METHODS})

    for metric in ALL_METRICS:
        mname = metric[0]
        base  = build_baseline(metric)
        raw_cache[mname]["dense"] = base
        for method in METHODS:
            raw_cache[mname][method] = {}
            for sp in SPARSITIES:
                pruned = build_pruned(metric, sp, method)
                raw_cache[mname][method][sp] = pruned
                common = sorted(set(base) & set(pruned))
                pct_cache[mname][method][sp] = {
                    l: abs(pruned[l] - base[l]) / abs(base[l]) * 100
                    for l in common if base[l] != 0
                }

    # canonical layer list (from dense core, which has all 26 layers)
    ALL_LAYERS = sorted(raw_cache["KL Div Score"]["dense"].keys())

    LAYER_GROUPS_3 = {
        "Early\n(0–8)":    [l for l in ALL_LAYERS if l <= 8],
        "Middle\n(9–17)":  [l for l in ALL_LAYERS if 9 <= l <= 17],
        "Late\n(18–25)":   [l for l in ALL_LAYERS if l >= 18],
    }

    # 5-layer windows: (label, layer list)
    _win_bounds = [(0,4),(5,9),(10,14),(15,19),(20,25)]
    LAYER_GROUPS_5 = {f"{a}–{b}": [l for l in ALL_LAYERS if a <= l <= b]
                    for a, b in _win_bounds}

    print("  Done.")

    # ── helpers for grouped aggregation ──────────────────────────────────────────

    def group_mean_pct(mname, method, sp, layers):
        """Mean % change for a method/sparsity over a list of layers."""
        vals = [pct_cache[mname][method][sp][l]
                for l in layers if l in pct_cache[mname][method].get(sp, {})]
        return float(np.mean(vals)) if vals else float("nan")



# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 5: — 5-layer window average line plots
#             All 7 metrics, sparsity=50.  x-axis = window label.
# ═══════════════════════════════════════════════════════════════════════════════
def exp5():
    print("=== Experiment 5: 5-layer window average lines ===")

    win_labels = list(LAYER_GROUPS_5.keys())
    x_win      = np.arange(len(win_labels))

    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
    fig.suptitle("% Change (5-layer window average),  sparsity=50%",
                fontsize=FONTDELTA+13, fontweight="bold")
    axes_flat = axes.flatten()

    for idx, metric in enumerate(ALL_METRICS):
        ax    = axes_flat[idx]
        mname = metric[0]
        for method in METHODS:
            y = [group_mean_pct(mname, method, 50, glayers)
                for glayers in LAYER_GROUPS_5.values()]
            ax.plot(x_win, y, label=method, **METHOD_STYLES[method],
                    linewidth=2, markersize=7)
        ax.set_xticks(x_win)
        ax.set_xticklabels(win_labels, fontsize=FONTDELTA+8, rotation=15)
        ax.set_ylabel("Mean % change", fontsize=FONTDELTA+10)
        ax.set_title(mname, fontsize=FONTDELTA+11, fontweight="semibold")
        ax.legend(fontsize=FONTDELTA+8, framealpha=0.7)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes_flat[-1].set_visible(False)
    outB = os.path.join(OUT_DIR, "varB_5layer_window.pdf")
    fig.savefig(outB, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outB}")

# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 6 — Heatmap: method × layer % change  (sparsity=50)
#
# 4×2 figure, one subplot per metric.
# Each subplot = 3 × N_layers heatmap (rows=methods, cols=layers).
# Color intensity shows which method × layer combinations hurt most.
# ═══════════════════════════════════════════════════════════════════════════════

def exp6():
    print("=== Experiment 6: Heatmap — Method × Layer ===")

    fig, axes = plt.subplots(4, 2, figsize=(18, 12), constrained_layout=True)
    fig.suptitle("% Change Heatmap: Methods × Layers  (sparsity=50%)",
                fontsize=FONTDELTA+20, fontweight="bold")
    axes_flat = axes.flatten()

    for idx, metric in enumerate(ALL_METRICS):
        ax    = axes_flat[idx]
        mname = metric[0]
        # Build matrix: rows = methods, cols = layers that have data for all methods
        all_present = set(ALL_LAYERS)
        for m in METHODS:
            all_present &= set(pct_cache[mname][m].get(50, {}).keys())
        cols = sorted(all_present)
        mat  = np.array([
            [pct_cache[mname][m][50].get(l, float("nan")) for l in cols]
            for m in METHODS
        ])
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=np.nanpercentile(mat, 95))
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(METHODS, fontsize=FONTDELTA+10)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=FONTDELTA+10, rotation=90)
        ax.set_title(mname, fontsize=FONTDELTA+12, fontweight="normal")
        plt.colorbar(im, ax=ax, shrink=0.8, label="% change")

    # axes_flat[-1].set_visible(False)
    out6 = os.path.join(OUT_DIR, "fig3.pdf")
    fig.savefig(out6, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out6}")




# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 7 — Layer Sensitivity Profile  (all sparsities combined)
#
# For each layer, compute the mean % change across all 3 methods and all 3
# sparsities → a single "sensitivity score" per layer.  Plot as a bar chart
# with color indicating early/mid/late grouping.
# Additionally, stack bars by sparsity to show how sensitivity grows.
# ═══════════════════════════════════════════════════════════════════════════════
def exp7():
    print("=== Experiment 7: Layer Sensitivity Profile ===")

    # Sensitivity matrix: [layer × sparsity], averaged over methods AND metrics
    n_layers  = len(ALL_LAYERS)
    n_sp      = len(SPARSITIES)
    sens_mat  = np.full((n_layers, n_sp), float("nan"))  # [layer, sp_idx]

    for li, layer in enumerate(ALL_LAYERS):
        for si, sp in enumerate(SPARSITIES):
            vals = []
            for metric in ALL_METRICS:
                mname = metric[0]
                for method in METHODS:
                    v = pct_cache[mname][method][sp].get(layer)
                    if v is not None:
                        vals.append(v)
            if vals:
                sens_mat[li, si] = np.mean(vals)

    # Group colors for bars
    def layer_group_color(layer):
        if layer <= 8:   return "#6A0572"
        if layer <= 17:  return "#1D6FA4"
        return "#E07A2F"

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    # fig, axes = plt.subplots(1, 1, figsize=(16, 10), constrained_layout=True)
    fig.suptitle("Layer Sensitivity Profile: Mean % change averaged across all methods and metrics",
                fontsize=FONTDELTA+20, fontweight="bold")

    # (a) Stacked bar: contribution per sparsity level
    ax  = axes[0]
    bot = np.zeros(n_layers)
    sp_colors = ["#A8DADC", "#457B9D", "#1D3557"]
    for si, sp in enumerate(SPARSITIES):
        inc = np.where(np.isnan(sens_mat[:, si]), 0, sens_mat[:, si])
        ax.bar(ALL_LAYERS, inc, bottom=bot, color=sp_colors[si],
            alpha=0.85, edgecolor="white", label=f"sparsity={sp}%")
        bot += inc
    ax.set_xlabel("Layer", fontsize=FONTDELTA+10)
    ax.set_ylabel("Mean % change (stacked by sparsity)", fontsize=FONTDELTA+10)
    ax.set_title("(a) Stacked by sparsity level", fontsize=FONTDELTA+12)
    ax.legend(fontsize=FONTDELTA+10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    # Shade early/mid/late regions
    ax.axvspan(-0.5,  8.5, alpha=0.06, color="#6A0572")
    ax.axvspan( 8.5, 17.5, alpha=0.06, color="#1D6FA4")
    ax.axvspan(17.5, 25.5, alpha=0.06, color="#E07A2F")
    for xl, lbl in [(4, "Early"), (13, "Middle"), (21.5, "Late")]:
        ax.text(xl, ax.get_ylim()[1] * 0.95, lbl, ha="center", fontsize=FONTDELTA+10,
                color="gray", style="italic")

    # (b) Single bar colored by group, with sp=50 only (clean, comparable)
    ax  = axes[1]
    vals50 = sens_mat[:, SPARSITIES.index(50)]
    colors = [layer_group_color(l) for l in ALL_LAYERS]
    ax.bar(ALL_LAYERS, np.where(np.isnan(vals50), 0, vals50),
        color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Layer", fontsize=FONTDELTA+10)
    ax.set_ylabel("Mean % change (sparsity=50%)", fontsize=FONTDELTA+10)
    ax.set_title("(b) sparsity=50% only, colored by Early/Middle/Late", fontsize=FONTDELTA+12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    from matplotlib.patches import Patch
    legend_handles = [Patch(color="#6A0572", alpha=0.85, label="Early (0–8)"),
                    Patch(color="#1D6FA4", alpha=0.85, label="Middle (9–17)"),
                    Patch(color="#E07A2F", alpha=0.85, label="Late (18–25)")]
    ax.legend(handles=legend_handles, fontsize=FONTDELTA+10)

    out7 = os.path.join(OUT_DIR, "fig4.pdf")
    fig.savefig(out7, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out7}")

# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 8 — SCR & TPP Threshold Comparison  (sparsity=50)
#
# For thresholds [2, 5, 10, 20, 50, 100, 500]:
#   (a) Mean raw metric value vs threshold  — dense + 3 methods
#   (b) Mean % change from dense vs threshold — 3 methods
# Layout: 2×2 figure (SCR row / TPP row  ×  raw / % change col).
# x-axis is log-scaled to space threshold values legibly.
# ═══════════════════════════════════════════════════════════════════════════════

def exp8():
    print("=== Experiment 8: SCR & TPP Threshold Comparison (sparsity=50) ===")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        "SCR & TPP Threshold Comparison  (sparsity=50%)\n"
        "Left: mean raw value vs threshold  |  Right: mean % change from dense",
        fontsize=FONTDELTA+20, fontweight="bold",
    )

    def _threshold_series(eval_type, key_fmt, thresholds, sparsity, method):
        """Return (raw_means, pct_means) lists over thresholds for one method."""
        raw_means, pct_means = [], []
        for thr in thresholds:
            keys   = key_fmt(thr)
            base   = load_layers(interp_glob("dense", None, eval_type), *keys)
            pruned = load_layers(interp_glob(sparsity, method, eval_type), *keys)
            raw_means.append(float(np.nanmean(list(pruned.values()))) if pruned else float("nan"))
            _, pct = pct_change_series(base, pruned)
            pct_means.append(float(np.nanmean(pct)) if pct else float("nan"))
        return raw_means, pct_means

    def _dense_means(eval_type, key_fmt, thresholds):
        return [
            float(np.nanmean(list(
                load_layers(interp_glob("dense", None, eval_type), *key_fmt(thr)).values()
            )))
            for thr in thresholds
        ]

    def _format_axes(ax_raw, ax_pct, thresholds, raw_label, title_prefix):
        for ax in (ax_raw, ax_pct):
            ax.set_xscale("log")
            ax.set_xticks(thresholds)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_xlabel("Threshold", fontsize=FONTDELTA+10)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(fontsize=FONTDELTA+10, framealpha=0.8)
        ax_raw.set_ylabel(raw_label, fontsize=FONTDELTA+10)
        ax_raw.set_title(f"{title_prefix}: Mean Raw Value vs Threshold",
                         fontsize=FONTDELTA+12, fontweight="normal")
        ax_pct.set_ylabel("Mean % change from dense", fontsize=FONTDELTA+10)
        ax_pct.set_title(f"{title_prefix}: % Change vs Threshold  (sparsity=50%)",
                         fontsize=FONTDELTA+12, fontweight="normal")

    # ── SCR ──────────────────────────────────────────────────────────────────
    ax_raw_scr, ax_pct_scr = axes[0, 0], axes[0, 1]
    scr_key = lambda thr: ("scr_metrics", f"scr_metric_threshold_{thr}")

    dense_scr = _dense_means("scr", scr_key, SCR_THRESHOLDS)
    ax_raw_scr.plot(SCR_THRESHOLDS, dense_scr, **DENSE_STYLE, label="dense")

    for method in METHODS:
        raw_m, pct_m = _threshold_series("scr", scr_key, SCR_THRESHOLDS, 50, method)
        ax_raw_scr.plot(SCR_THRESHOLDS, raw_m, label=method,
                        **METHOD_STYLES[method], linewidth=1.8, markersize=6)
        ax_pct_scr.plot(SCR_THRESHOLDS, pct_m, label=method,
                        **METHOD_STYLES[method], linewidth=1.8, markersize=6)

    _format_axes(ax_raw_scr, ax_pct_scr, SCR_THRESHOLDS,
                 "Mean SCR metric (raw)", "SCR")

    # ── TPP ──────────────────────────────────────────────────────────────────
    ax_raw_tpp, ax_pct_tpp = axes[1, 0], axes[1, 1]
    tpp_key = lambda thr: ("tpp_metrics", f"tpp_threshold_{thr}_total_metric")

    dense_tpp = _dense_means("tpp", tpp_key, TPP_THRESHOLDS)
    ax_raw_tpp.plot(TPP_THRESHOLDS, dense_tpp, **DENSE_STYLE, label="dense")

    for method in METHODS:
        raw_m, pct_m = _threshold_series("tpp", tpp_key, TPP_THRESHOLDS, 50, method)
        ax_raw_tpp.plot(TPP_THRESHOLDS, raw_m, label=method,
                        **METHOD_STYLES[method], linewidth=1.8, markersize=6)
        ax_pct_tpp.plot(TPP_THRESHOLDS, pct_m, label=method,
                        **METHOD_STYLES[method], linewidth=1.8, markersize=6)

    _format_axes(ax_raw_tpp, ax_pct_tpp, TPP_THRESHOLDS,
                 "Mean TPP total metric (raw)", "TPP")

    out8 = os.path.join(OUT_DIR, "appfig3.pdf")
    fig.savefig(out8, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out8}")


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 9 — Ranking stability: which method is best at each threshold?
#
# For each (SCR / TPP), at every threshold compute mean % change per method,
# rank them 1 (best) → 3 (worst), then plot rank vs threshold.
# A flat line = stable winner; crossing lines = threshold-dependent conclusions.
# ═══════════════════════════════════════════════════════════════════════════════

def exp9():
    print("=== Experiment 9: Ranking Stability across Thresholds ===")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(
        "Method Ranking Stability vs Threshold  (sparsity=50%)\n"
        "Rank 1 = lowest % change from dense (best preserved);  "
        "crossing lines = threshold-sensitive conclusions",
        fontsize=FONTDELTA+20, fontweight="bold",
    )

    configs = [
        ("scr", lambda thr: ("scr_metrics", f"scr_metric_threshold_{thr}"),
         SCR_THRESHOLDS, "SCR"),
        ("tpp", lambda thr: ("tpp_metrics", f"tpp_threshold_{thr}_total_metric"),
         TPP_THRESHOLDS, "TPP"),
    ]

    for ax, (eval_type, key_fmt, thresholds, title) in zip(axes, configs):
        ranks = {m: [] for m in METHODS}
        for thr in thresholds:
            keys = key_fmt(thr)
            base = load_layers(interp_glob("dense", None, eval_type), *keys)
            pct_means = {}
            for method in METHODS:
                pruned = load_layers(interp_glob(50, method, eval_type), *keys)
                _, pct = pct_change_series(base, pruned)
                pct_means[method] = float(np.nanmean(pct)) if pct else float("nan")
            for rank, method in enumerate(
                sorted(METHODS, key=lambda m: pct_means[m]), start=1
            ):
                ranks[method].append(rank)

        for method in METHODS:
            ax.plot(thresholds, ranks[method], label=method,
                    **METHOD_STYLES[method], linewidth=2, markersize=8)

        ax.set_xscale("log")
        ax.set_xticks(thresholds)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["1st\n(best)", "2nd", "3rd\n(worst)"], fontsize=FONTDELTA+10)
        ax.invert_yaxis()
        ax.set_xlabel("Threshold", fontsize=FONTDELTA+10)
        ax.set_title(f"{title}: Ranking vs Threshold", fontsize=FONTDELTA+12, fontweight="normal")
        ax.legend(fontsize=FONTDELTA+10, framealpha=0.8)
        ax.grid(True, linestyle=":", alpha=0.5)

    out = os.path.join(OUT_DIR, "exp9_ranking_stability.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 10 — Cross-Metric Correlation Matrix  (validates failure-mode orthogonality)
#
# For each (method, layer) at sparsity=50, collect all 8 metric % changes.
# Compute pairwise Pearson correlations across the (method × layer) sample.
# Theory predicts: Core metrics cluster (failure mode 1), Absorption separate
# (failure mode 2), SCR/TPP cluster together (failure mode 3).
# Boxes highlight the predicted failure-mode groups.
# ═══════════════════════════════════════════════════════════════════════════════

def exp10():
    print("=== Experiment 10: Cross-Metric Correlation Matrix ===")

    metric_names = [m[0] for m in ALL_METRICS]
    n_m = len(metric_names)

    # Build data matrix: rows = (method, layer), cols = metrics
    rows = []
    for method in METHODS:
        for layer in ALL_LAYERS:
            row = [pct_cache[mn][method][50].get(layer, float("nan"))
                   for mn in metric_names]
            rows.append(row)
    data = np.array(rows, dtype=float)  # shape (n_methods*n_layers, n_metrics)

    # Pairwise Pearson correlation, handling NaNs per pair
    corr = np.full((n_m, n_m), float("nan"))
    for i in range(n_m):
        for j in range(n_m):
            mask = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
            if mask.sum() > 2:
                corr[i, j] = np.corrcoef(data[mask, i], data[mask, j])[0, 1]

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    fig.suptitle(
        "Cross-Metric Correlation of % Degradation  (sparsity=50%)\n"
        "Each cell = Pearson r across all (method × layer) pairs\n"
        "Boxes = predicted failure-mode groups (Core / Absorption / SCR+TPP)",
        fontsize=FONTDELTA+12, fontweight="bold",
    )

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_m))
    ax.set_xticklabels(metric_names, fontsize=FONTDELTA+9, rotation=30, ha="right")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(metric_names, fontsize=FONTDELTA+9)

    for i in range(n_m):
        for j in range(n_m):
            v = corr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FONTDELTA+8, color="black" if abs(v) < 0.6 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    # Draw predicted failure-mode group boxes
    from matplotlib.patches import Rectangle
    group_boxes = [
        (0, 4, "#E63946", "Failure mode 1\n(Core)"),
        (5, 5, "#457B9D", "Failure mode 2\n(Absorption)"),
        (6, 7, "#2A9D8F", "Failure mode 3\n(SCR / TPP)"),
    ]
    for start, end, color, _ in group_boxes:
        size = end - start + 1
        rect = Rectangle((start - 0.5, start - 0.5), size, size,
                          fill=False, edgecolor=color, linewidth=2.5, linestyle="-")
        ax.add_patch(rect)

    # Legend for boxes
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=c, linewidth=2.5, label=lbl)
        for _, _, c, lbl in group_boxes
    ]
    ax.legend(handles=legend_handles, fontsize=FONTDELTA+8, framealpha=0.85,
              loc="upper left", bbox_to_anchor=(1.18, 1.0))

    out = os.path.join(OUT_DIR, "appfig4.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

# ── Summary ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # # exp1()
    # exp1_2_combined()
    # # exp2()
    # exp3_1()
    # exp3_2()
    # exp4()
    helpV()  # pre-load data cache for extra experiments
    # # exp5()      
    # exp6()
    exp7()
    # exp8()
    # # exp9()
    # exp10()
    print("\nAll experiments completed and figures saved.")
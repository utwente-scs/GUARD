"""
Generate all thesis figures for GUARD extension results.

Usage:
    python generate_figures.py

Outputs PNG files to results/figures/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Output directory
OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Color palette — colorblind-friendly, works in print
C_BLUE = "#2166ac"
C_RED = "#b2182b"
C_GREEN = "#1b7837"
C_ORANGE = "#e08214"
C_PURPLE = "#6a3d9a"
C_GREY = "#878787"
C_LIGHT_BLUE = "#92c5de"
C_LIGHT_RED = "#fddbc7"


def add_bar_labels(ax, bars, fmt=".3f", offset=0.02):
    """Add value labels above bars."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    f"{h:{fmt}}", ha="center", va="bottom", fontsize=8)


# =========================================================
# Figure 1: RQ1 → RQ2 Progression (P, R, F1 per language)
# =========================================================
def fig1_progression():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    languages = ["Puppet", "Terraform", "Chef"]

    # [Baseline, Extended (Ansible-only), Best retrained]
    data = {
        "Puppet": {
            "Precision": [0.035, 0.228, 0.272],
            "Recall":    [0.093, 0.852, 0.796],
            "F1":        [0.051, 0.359, 0.406],
        },
        "Terraform": {
            "Precision": [0.000, 0.452, None],
            "Recall":    [0.000, 0.852, None],
            "F1":        [0.000, 0.591, None],
        },
        "Chef": {
            "Precision": [0.000, 0.309, 0.432],
            "Recall":    [0.000, 0.613, 0.640],
            "F1":        [0.000, 0.411, 0.516],
        },
    }

    phases_3 = ["Baseline\n(RQ1)", "Parser\nextension", "Per-language\nretraining"]
    phases_2 = ["Baseline\n(RQ1)", "Parser\nextension"]
    colors = [C_GREY, C_BLUE, C_GREEN]
    metrics = ["Precision", "Recall", "F1"]
    hatches = ["", "//", ""]

    for idx, lang in enumerate(languages):
        ax = axes[idx]
        d = data[lang]
        is_terraform = lang == "Terraform"
        n_phases = 2 if is_terraform else 3
        phases = phases_2 if is_terraform else phases_3

        x = np.arange(n_phases)
        width = 0.25

        for i, metric in enumerate(metrics):
            vals = d[metric][:n_phases]
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          label=metric if idx == 0 else "",
                          color=colors[i], alpha=0.85)
            add_bar_labels(ax, bars)

        ax.set_title(lang)
        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_ylim(0, 1.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if is_terraform:
            ax.annotate("No retraining\n(taxonomy mismatch)",
                        xy=(1.3, 0.7), fontsize=8, fontstyle="italic",
                        ha="center", color=C_GREY)

    axes[0].set_ylabel("Score")
    fig.legend(metrics, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Detection Performance: Baseline → Extension → Retraining",
                 y=1.08, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_progression.png")
    plt.close()
    print(f"  Saved fig1_progression.png")


# =========================================================
# Figure 2: Puppet Retraining Comparison (4 configs)
# =========================================================
def fig2_puppet_configs():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    configs = ["Ansible-only", "Combined", "Puppet+Ansible", "Puppet-only"]
    tp = [46, 51, 43, 43]
    fp = [156, 145, 240, 115]
    fn = [8, 3, 11, 11]
    precision = [0.228, 0.260, 0.152, 0.272]
    recall = [0.852, 0.944, 0.796, 0.796]
    f1 = [0.359, 0.408, 0.255, 0.406]
    config_colors = [C_BLUE, C_ORANGE, C_RED, C_GREEN]

    # Left: P/R/F1
    x = np.arange(len(configs))
    w = 0.25
    b1 = ax1.bar(x - w, precision, w, label="Precision", color=C_GREY, alpha=0.85)
    b2 = ax1.bar(x, recall, w, label="Recall", color=C_BLUE, alpha=0.85)
    b3 = ax1.bar(x + w, f1, w, label="F1", color=C_GREEN, alpha=0.85)
    add_bar_labels(ax1, b1)
    add_bar_labels(ax1, b2)
    add_bar_labels(ax1, b3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=15, ha="right")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Metrics by Configuration")
    ax1.legend(frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: TP/FP raw counts (stacked)
    bars_tp = ax2.bar(x, tp, 0.5, label="True Positives", color=C_GREEN, alpha=0.85)
    bars_fp = ax2.bar(x, fp, 0.5, bottom=tp, label="False Positives", color=C_RED, alpha=0.6)
    for i, (t, f) in enumerate(zip(tp, fp)):
        ax2.text(i, t / 2, str(t), ha="center", va="center", fontsize=9,
                 fontweight="bold", color="white")
        ax2.text(i, t + f / 2, str(f), ha="center", va="center", fontsize=9,
                 fontweight="bold", color="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=15, ha="right")
    ax2.set_ylabel("Count")
    ax2.set_title("TP vs FP: Raw Counts")
    ax2.legend(frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Puppet — Retraining Strategy Comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_puppet_configs.png")
    plt.close()
    print(f"  Saved fig2_puppet_configs.png")


# =========================================================
# Figure 3: Chef Retraining Comparison (3 configs)
# =========================================================
def fig3_chef_configs():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    configs = ["Ansible-only", "Combined", "Chef-only"]
    tp = [46, 48, 48]
    fp = [103, 147, 63]
    fn = [29, 27, 27]
    precision = [0.309, 0.246, 0.432]
    recall = [0.613, 0.640, 0.640]
    f1 = [0.411, 0.356, 0.516]

    x = np.arange(len(configs))
    w = 0.25

    b1 = ax1.bar(x - w, precision, w, label="Precision", color=C_GREY, alpha=0.85)
    b2 = ax1.bar(x, recall, w, label="Recall", color=C_BLUE, alpha=0.85)
    b3 = ax1.bar(x + w, f1, w, label="F1", color=C_GREEN, alpha=0.85)
    add_bar_labels(ax1, b1)
    add_bar_labels(ax1, b2)
    add_bar_labels(ax1, b3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Metrics by Configuration")
    ax1.legend(frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    bars_tp = ax2.bar(x, tp, 0.5, label="True Positives", color=C_GREEN, alpha=0.85)
    bars_fp = ax2.bar(x, fp, 0.5, bottom=tp, label="False Positives", color=C_RED, alpha=0.6)
    for i, (t, f) in enumerate(zip(tp, fp)):
        ax2.text(i, t / 2, str(t), ha="center", va="center", fontsize=9,
                 fontweight="bold", color="white")
        ax2.text(i, t + f / 2, str(f), ha="center", va="center", fontsize=9,
                 fontweight="bold", color="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.set_ylabel("Count")
    ax2.set_title("TP vs FP: Raw Counts")
    ax2.legend(frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Chef — Retraining Strategy Comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_chef_configs.png")
    plt.close()
    print(f"  Saved fig3_chef_configs.png")


# =========================================================
# Figure 4: GUARD vs GLITCH — Side by Side
# =========================================================
def fig4_guard_vs_glitch():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    metrics = ["Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    w = 0.3

    # Chef
    glitch_chef = [0.5732, 0.6267, 0.5987]
    guard_chef = [0.4324, 0.6400, 0.5161]

    b1 = ax1.bar(x - w/2, glitch_chef, w, label="GLITCH", color=C_BLUE, alpha=0.85)
    b2 = ax1.bar(x + w/2, guard_chef, w, label="GUARD", color=C_GREEN, alpha=0.85)
    add_bar_labels(ax1, b1)
    add_bar_labels(ax1, b2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Chef")
    ax1.legend(frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Puppet
    glitch_puppet = [0.4425, 0.9259, 0.5988]
    guard_puppet = [0.2602, 0.9444, 0.4080]

    b1 = ax2.bar(x - w/2, glitch_puppet, w, label="GLITCH", color=C_BLUE, alpha=0.85)
    b2 = ax2.bar(x + w/2, guard_puppet, w, label="GUARD", color=C_GREEN, alpha=0.85)
    add_bar_labels(ax2, b1)
    add_bar_labels(ax2, b2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1.12)
    ax2.set_title("Puppet")
    ax2.legend(frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("RQ3: GUARD vs GLITCH (GUARD-compatible smells only)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_guard_vs_glitch.png")
    plt.close()
    print(f"  Saved fig4_guard_vs_glitch.png")


# =========================================================
# Figure 5: Detection Overlap (RQ3)
# =========================================================
def fig5_overlap():
    from matplotlib_venn import venn2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Chef: GLITCH=47 TP, GUARD=48 TP, shared=46
    v1 = venn2(subsets=(1, 2, 46), set_labels=("GLITCH", "GUARD"), ax=ax1,
               set_colors=(C_BLUE, C_GREEN), alpha=0.6)
    v1.get_label_by_id("10").set_text("1")
    v1.get_label_by_id("01").set_text("2")
    v1.get_label_by_id("11").set_text("46")
    ax1.set_title("Chef — True Positive Overlap\n(96% shared)", fontweight="bold")

    # Puppet: GLITCH=50 TP, GUARD=51 TP, shared=48
    v2 = venn2(subsets=(2, 3, 48), set_labels=("GLITCH", "GUARD"), ax=ax2,
               set_colors=(C_BLUE, C_GREEN), alpha=0.6)
    v2.get_label_by_id("10").set_text("2")
    v2.get_label_by_id("01").set_text("3")
    v2.get_label_by_id("11").set_text("48")
    ax2.set_title("Puppet — True Positive Overlap\n(94% shared)", fontweight="bold")

    fig.suptitle("RQ3: Detection Overlap — GUARD vs GLITCH", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_overlap.png")
    plt.close()
    print(f"  Saved fig5_overlap.png")


# =========================================================
# Figure 6: Training Data Quality → Precision
# =========================================================
def fig6_data_quality():
    fig, ax = plt.subplots(figsize=(7, 5))

    # Data points: (training data accuracy, best precision achieved, label)
    points = [
        (0.46, 0.272, "Puppet\n(1,332 samples)"),
        (0.79, 0.432, "Chef\n(2,106 samples)"),
    ]

    for acc, prec, label in points:
        ax.scatter(acc, prec, s=200, zorder=5, color=C_BLUE, edgecolors="black",
                   linewidth=1)
        ax.annotate(label, (acc, prec), textcoords="offset points",
                    xytext=(15, -5), fontsize=10, va="center")

    # Trend line
    accs = [p[0] for p in points]
    precs = [p[1] for p in points]
    z = np.polyfit(accs, precs, 1)
    x_line = np.linspace(0.3, 1.0, 100)
    y_line = np.polyval(z, x_line)
    ax.plot(x_line, y_line, "--", color=C_GREY, alpha=0.5, linewidth=1)

    # Reference: GLITCH precision
    ax.axhline(y=0.573, color=C_RED, linestyle=":", alpha=0.6, linewidth=1)
    ax.text(0.95, 0.583, "GLITCH Chef precision", ha="right", fontsize=8,
            color=C_RED, fontstyle="italic")
    ax.axhline(y=0.443, color=C_RED, linestyle=":", alpha=0.6, linewidth=1)
    ax.text(0.95, 0.453, "GLITCH Puppet precision", ha="right", fontsize=8,
            color=C_RED, fontstyle="italic")

    # Projection zone
    ax.axvspan(0.90, 1.0, alpha=0.08, color=C_GREEN)
    ax.text(0.95, 0.32, "Clean labels\n(projected)", ha="center", fontsize=8,
            color=C_GREEN, fontstyle="italic")

    ax.set_xlabel("Training Data Accuracy (verified)")
    ax.set_ylabel("Best Precision Achieved")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.15, 0.65)
    ax.set_title("Training Data Quality vs Detection Precision",
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_data_quality.png")
    plt.close()
    print(f"  Saved fig6_data_quality.png")


# =========================================================
# Figure 7: GUARD vs GLITCH — TP/FP Raw Counts
# =========================================================
def fig7_tp_fp_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    tools = ["GLITCH", "GUARD"]
    x = np.arange(len(tools))
    w = 0.5

    # Chef
    tp_chef = [47, 48]
    fp_chef = [35, 63]
    b_tp = ax1.bar(x, tp_chef, w, label="True Positives", color=C_GREEN, alpha=0.85)
    b_fp = ax1.bar(x, fp_chef, w, bottom=tp_chef, label="False Positives",
                   color=C_RED, alpha=0.6)
    for i, (t, f) in enumerate(zip(tp_chef, fp_chef)):
        ax1.text(i, t / 2, f"TP: {t}", ha="center", va="center", fontsize=10,
                 fontweight="bold", color="white")
        ax1.text(i, t + f / 2, f"FP: {f}", ha="center", va="center", fontsize=10,
                 fontweight="bold", color="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tools)
    ax1.set_ylabel("Count")
    ax1.set_title("Chef")
    ax1.legend(frameon=False, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Puppet
    tp_pup = [50, 51]
    fp_pup = [63, 145]
    b_tp = ax2.bar(x, tp_pup, w, label="True Positives", color=C_GREEN, alpha=0.85)
    b_fp = ax2.bar(x, fp_pup, w, bottom=tp_pup, label="False Positives",
                   color=C_RED, alpha=0.6)
    for i, (t, f) in enumerate(zip(tp_pup, fp_pup)):
        ax2.text(i, t / 2, f"TP: {t}", ha="center", va="center", fontsize=10,
                 fontweight="bold", color="white")
        ax2.text(i, t + f / 2, f"FP: {f}", ha="center", va="center", fontsize=10,
                 fontweight="bold", color="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tools)
    ax2.set_title("Puppet")
    ax2.legend(frameon=False, loc="upper left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("RQ3: GUARD vs GLITCH — Detection vs Noise",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_tp_fp_comparison.png")
    plt.close()
    print(f"  Saved fig7_tp_fp_comparison.png")


# =========================================================
# Figure 8: Terraform Taxonomy Breakdown
# =========================================================
def fig8_terraform_taxonomy():
    fig, ax = plt.subplots(figsize=(7, 5))

    categories = [
        "Missing attribute\n(no logging, encryption,\nIAM, etc.)",
        "GUARD-compatible\n(hardcoded secrets, HTTP,\nIP binding, etc.)",
        "Other non-GUARD"
    ]
    counts = [291, 66, 19]  # 376 total: 66 GUARD, 291 missing-attr, 19 other
    colors_pie = [C_RED, C_GREEN, C_GREY]
    explode = (0.03, 0.08, 0.03)

    wedges, texts, autotexts = ax.pie(
        counts, labels=categories, colors=colors_pie, explode=explode,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * sum(counts) / 100))})",
        startangle=140, textprops={"fontsize": 9}, pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    ax.set_title("Terraform Oracle — Smell Distribution\n"
                 "(376 total annotated smells)",
                 fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig8_terraform_taxonomy.png")
    plt.close()
    print(f"  Saved fig8_terraform_taxonomy.png")


# =========================================================
# Figure 9: Improvement Waterfall (Baseline → Parser → ML)
# =========================================================
def fig9_improvement_waterfall():
    """Shows how much each phase contributed to the final F1."""
    fig, ax = plt.subplots(figsize=(10, 5))

    languages = ["Puppet", "Chef"]
    baseline_f1 = [0.051, 0.000]
    parser_f1 = [0.359, 0.411]
    final_f1 = [0.406, 0.516]

    parser_gain = [p - b for p, b in zip(parser_f1, baseline_f1)]
    ml_gain = [f - p for f, p in zip(final_f1, parser_f1)]

    x = np.arange(len(languages))
    w = 0.45

    # Stacked: baseline + parser gain + ML gain
    b1 = ax.bar(x, baseline_f1, w, label="Baseline F1", color=C_GREY, alpha=0.7)
    b2 = ax.bar(x, parser_gain, w, bottom=baseline_f1,
                label="Parser extension gain", color=C_BLUE, alpha=0.85)
    b3 = ax.bar(x, ml_gain, w,
                bottom=[b + p for b, p in zip(baseline_f1, parser_gain)],
                label="ML retraining gain", color=C_GREEN, alpha=0.85)

    # Labels inside bars
    for i in range(len(languages)):
        if baseline_f1[i] > 0.02:
            ax.text(i, baseline_f1[i] / 2, f"{baseline_f1[i]:.3f}",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold")
        # Parser gain
        mid = baseline_f1[i] + parser_gain[i] / 2
        ax.text(i, mid, f"+{parser_gain[i]:.3f}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        # ML gain
        mid2 = baseline_f1[i] + parser_gain[i] + ml_gain[i] / 2
        ax.text(i, mid2, f"+{ml_gain[i]:.3f}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        # Total on top
        ax.text(i, final_f1[i] + 0.02, f"F1 = {final_f1[i]:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Percentage annotations
    for i in range(len(languages)):
        total_gain = final_f1[i] - baseline_f1[i]
        if total_gain > 0:
            parser_pct = parser_gain[i] / total_gain * 100
            ml_pct = ml_gain[i] / total_gain * 100
            ax.annotate(f"Parser: {parser_pct:.0f}%\nML: {ml_pct:.0f}%",
                        xy=(i + 0.3, final_f1[i] - 0.05),
                        fontsize=8, fontstyle="italic", color=C_GREY)

    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 0.65)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("F1 Improvement Breakdown: Parser vs ML Contribution",
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig9_improvement_waterfall.png")
    plt.close()
    print(f"  Saved fig9_improvement_waterfall.png")


# =========================================================
# Figure 10: Precision-Recall Trade-off (Puppet configs)
# =========================================================
def fig10_pr_tradeoff():
    """Scatter plot showing the P-R trade-off across all Puppet configs."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    configs = {
        "Ansible-only":    (0.852, 0.228, 0.359, 156, C_BLUE),
        "Combined":        (0.944, 0.260, 0.408, 145, C_ORANGE),
        "Puppet+Ansible":  (0.796, 0.152, 0.255, 240, C_RED),
        "Puppet-only":     (0.796, 0.272, 0.406, 115, C_GREEN),
    }

    for name, (r, p, f1, fp, color) in configs.items():
        size = fp * 2  # bubble size = FP count
        ax.scatter(r, p, s=size, color=color, alpha=0.6, edgecolors="black",
                   linewidth=1, zorder=5)
        ax.annotate(f"{name}\nF1={f1:.3f}, FP={fp}",
                    (r, p), textcoords="offset points",
                    xytext=(12, 8), fontsize=8, va="bottom")

    # F1 iso-curves
    for f1_val in [0.2, 0.3, 0.4, 0.5]:
        r_range = np.linspace(0.01, 1.0, 200)
        p_from_f1 = (f1_val * r_range) / (2 * r_range - f1_val)
        mask = (p_from_f1 > 0) & (p_from_f1 <= 1.0)
        ax.plot(r_range[mask], p_from_f1[mask], "--", color=C_GREY, alpha=0.3,
                linewidth=0.8)
        # Label on the right edge
        idx = np.argmin(np.abs(r_range[mask] - 0.98))
        if len(p_from_f1[mask]) > idx:
            ax.text(0.98, p_from_f1[mask][idx], f"F1={f1_val}",
                    fontsize=7, color=C_GREY, va="bottom")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.7, 1.02)
    ax.set_ylim(0.05, 0.4)
    ax.set_title("Puppet — Precision vs Recall Trade-off\n"
                 "(bubble size = false positive count)",
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig10_pr_tradeoff.png")
    plt.close()
    print(f"  Saved fig10_pr_tradeoff.png")


# =========================================================
# Run all
# =========================================================
if __name__ == "__main__":
    print("Generating figures...\n")
    fig1_progression()
    fig2_puppet_configs()
    fig3_chef_configs()
    fig4_guard_vs_glitch()
    fig5_overlap()
    fig6_data_quality()
    fig7_tp_fp_comparison()
    fig8_terraform_taxonomy()
    fig9_improvement_waterfall()
    fig10_pr_tradeoff()
    print(f"\nDone. All figures saved to {OUT}/")
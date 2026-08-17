
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

GUARD_COLOR = "#C03D3E"   
OTHER_COLOR = "#96A9C2"   
TEXT_COLOR = "#262626"

METRICS = [("f1", "F1"), ("precision", "PRECISION"), ("recall", "RECALL")]


def setup_style():
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": False,
    })


def panel(ax, df, metric, title, n_runs):
    k = df["n_removed"]
    for system, color, label in (("guard", GUARD_COLOR, "GUARD"),
                                 ("glitch", OTHER_COLOR, "GLITCH")):
        mean = df[f"{system}_{metric}_mean"]
        sd = df[f"{system}_{metric}_std"]
        ax.fill_between(k, (mean - sd).clip(0, 1), (mean + sd).clip(0, 1),
                        color=color, alpha=0.25, linewidth=0)
        ax.plot(k, mean, color=color, linewidth=1.8, label=label,
                zorder=3 if system == "guard" else 2)

    ax.set_title(f"{title} by keywords removed (n={n_runs} subsets per budget)")
    ax.set_xlabel("keywords removed")
    ax.set_ylabel(metric) 
    ax.set_xlim(0, int(k.max()))
    ax.set_ylim(0, 1)

    leg = ax.legend(frameon=False, loc="lower left")
    
    for text in leg.get_texts():
        if text.get_text() == "GUARD":
            text.set_color(GUARD_COLOR)
            text.set_fontweight("bold")

    sns.despine(ax=ax, offset=8, trim=False)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../../resources/out/ablation_regex.csv"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "../../resources/out")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    n_runs = int(df["runs"].iloc[0])
    setup_style()

    # --- stacked three-panel figure, same shape as Figure 1 ---
    fig, axes = plt.subplots(3, 1, figsize=(6.0, 11.0))
    for ax, (metric, title) in zip(axes, METRICS):
        panel(ax, df, metric, title, n_runs)
    fig.tight_layout(h_pad=3.0)
    fig.savefig(outdir / "fig_ablation_stacked.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_ablation_stacked.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # --- individual panels, in case only one metric earns the space ---
    for metric, title in METRICS:
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        panel(ax, df, metric, title, n_runs)
        fig.tight_layout()
        fig.savefig(outdir / f"fig_ablation_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"written to {outdir}")


if __name__ == "__main__":
    main()
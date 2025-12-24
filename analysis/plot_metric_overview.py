"""Generate publication-style composite overview of metric coverage."""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "figures" / "metric_overview.png"


def main() -> None:
    # Data preparation (matching provided reference)
    data_app = {
        "Theme": [
            "Federated Learning",
            "Model Performance",
            "Drift & Lifecycle",
            "Safety-Linked Thresholds",
            "Data Quality & Coverage",
            "Robustness & Stability",
            "Verification & Validation",
            "Bias & Fairness",
            "Runtime & Deployment",
            "Statistical Uncertainty",
            "Explainability",
        ],
        "Count": [9, 5, 4, 4, 3, 3, 3, 2, 2, 1, 1],
    }
    df_app = pd.DataFrame(data_app).sort_values("Count", ascending=True)

    data_exc = {
        "Theme": [
            "Process & Documentation",
            "Process & Lifecycle Mgmt",
            "Certification Process",
            "Human Factors Evaluation",
            "Operational Deployment",
            "HIL Testing",
        ],
        "Count": [18, 7, 4, 4, 4, 1],
    }
    df_exc = pd.DataFrame(data_exc).sort_values("Count", ascending=True)

    data_thresh = {
        "Source": [
            "Academic Literature",
            "Regulation (EASA/FAA)",
            "Industrial Best Practice",
        ],
        "Count": [22, 9, 7],
    }
    df_thresh = pd.DataFrame(data_thresh)

    # Professional style configuration
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["text.color"] = "#333333"
    plt.rcParams["axes.labelcolor"] = "#333333"
    plt.rcParams["xtick.color"] = "#333333"
    plt.rcParams["ytick.color"] = "#333333"

    colors_app = sns.color_palette("YlGnBu_r", len(df_app))
    colors_exc = sns.color_palette("Greys_r", len(df_exc))
    colors_pie = ["#2C5F2D", "#97BC62", "#A03C78"]

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1], figure=fig)
    fig.patch.set_facecolor("white")

    # Panel 1
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.barh(
        df_app["Theme"],
        df_app["Count"],
        color=colors_app,
        edgecolor="none",
        height=0.6,
    )
    ax1.set_title(
        "A. Quantitative / Experimental Metrics (N=38)\n(Operationalized in Benchmarking)",
        fontweight="bold",
        loc="left",
        pad=20,
    )
    ax1.set_xlabel("Number of Metrics")
    for spine in ["top", "right", "left"]:
        ax1.spines[spine].set_visible(False)
    ax1.xaxis.set_ticks_position("bottom")
    ax1.yaxis.set_ticks_position("none")
    ax1.grid(axis="x", linestyle=":", color="#CCCCCC")
    for bar in bars1:
        width = bar.get_width()
        ax1.text(
            width + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=bar.get_facecolor(),
        )

    # Panel 2
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.barh(
        df_exc["Theme"],
        df_exc["Count"],
        color=colors_exc,
        edgecolor="none",
        height=0.6,
    )
    ax2.set_title(
        "B. Qualitative / Process Metrics (N=38)\n(Required for Certification, Not Benchmarked)",
        fontweight="bold",
        loc="left",
        pad=20,
    )
    ax2.set_xlabel("Number of Metrics")
    for spine in ["top", "right", "left"]:
        ax2.spines[spine].set_visible(False)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.yaxis.set_ticks_position("none")
    ax2.grid(axis="x", linestyle=":", color="#CCCCCC")
    for bar in bars2:
        width = bar.get_width()
        ax2.text(
            width + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#555555",
        )

    # Panel 3
    ax3 = fig.add_subplot(gs[1, :])
    wedges, texts, autotexts = ax3.pie(
        df_thresh["Count"],
        labels=df_thresh["Source"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors_pie,
        pctdistance=0.8,
        textprops={"fontsize": 12},
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_weight("bold")
    ax3.text(
        0,
        0,
        "Threshold\nSources\n(N=38)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#333333",
    )
    ax3.set_title(
        "C. Origin of Acceptance Thresholds (for Quantitative Metrics in A)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.05, hspace=0.5, wspace=0.4)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()


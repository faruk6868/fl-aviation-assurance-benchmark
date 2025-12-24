"""Generate Sankey diagram for assurance metric framework."""

from __future__ import annotations

from pathlib import Path

from packaging.version import InvalidVersion, Version
import plotly
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"


def build_nodes() -> list[str]:
    base_nodes = [
        "Total Metrics (N=76)",
        "Quantitative (App.) (N=38)",
        "Qualitative (Exc.) (N=38)",
        "Federated Learning (N=9)",
        "Model Performance (N=5)",
        "Drift & Lifecycle (N=4)",
        "Safety-Linked Thresholds (N=4)",
        "Data Quality & Coverage (N=3)",
        "Robustness & Stability (N=3)",
        "Verification & Validation (N=3)",
        "Bias & Fairness (N=2)",
        "Runtime & Deployment (N=2)",
        "Statistical Uncertainty (N=1)",
        "Explainability (N=1)",
        "Process & Documentation (N=18)",
        "Process & Lifecycle Mgmt (N=7)",
        "Certification Process (N=4)",
        "Human Factors Eval. (N=4)",
        "Operational Deployment (N=4)",
        "HIL Testing (N=1)",
        "Academic Source (N=22)",
        "Regulation Source (N=9)",
        "Industrial Source (N=7)",
    ]
    base_nodes.append("Thresholds Needed (N=38)")
    return base_nodes


def build_links(agg_idx: int) -> list[dict[str, object]]:
    links: list[dict[str, object]] = [
        {"source": 0, "target": 1, "value": 38, "color": "rgba(70,130,180,0.4)"},
        {"source": 0, "target": 2, "value": 38, "color": "rgba(169,169,169,0.4)"},
        # Quantitative themes
        {"source": 1, "target": 3, "value": 9},
        {"source": 1, "target": 4, "value": 5},
        {"source": 1, "target": 5, "value": 4},
        {"source": 1, "target": 6, "value": 4},
        {"source": 1, "target": 7, "value": 3},
        {"source": 1, "target": 8, "value": 3},
        {"source": 1, "target": 9, "value": 3},
        {"source": 1, "target": 10, "value": 2},
        {"source": 1, "target": 11, "value": 2},
        {"source": 1, "target": 12, "value": 1},
        {"source": 1, "target": 13, "value": 1},
        # Qualitative themes
        {"source": 2, "target": 14, "value": 18},
        {"source": 2, "target": 15, "value": 7},
        {"source": 2, "target": 16, "value": 4},
        {"source": 2, "target": 17, "value": 4},
        {"source": 2, "target": 18, "value": 4},
        {"source": 2, "target": 19, "value": 1},
    ]

    quant_theme_indices = range(3, 14)
    for idx in quant_theme_indices:
        theme_value = next(link["value"] for link in links if link["target"] == idx)
        links.append(
            {
                "source": idx,
                "target": agg_idx,
                "value": theme_value,
                "color": "rgba(70,130,180,0.25)",
            }
        )

    links.extend(
        [
            {"source": agg_idx, "target": 20, "value": 22, "color": "rgba(44,95,45,0.5)"},
            {"source": agg_idx, "target": 21, "value": 9, "color": "rgba(160,60,120,0.5)"},
            {"source": agg_idx, "target": 22, "value": 7, "color": "rgba(151,188,98,0.5)"},
        ]
    )
    return links


def build_colors(agg_idx: int, node_count: int) -> list[str]:
    colors = [
        "#333333",
        "#4682B4",
        "#A9A9A9",
    ]
    colors.extend(["#87CEEB"] * 11)
    colors.extend(["#D3D3D3"] * 6)
    colors.extend(["#2C5F2D", "#A03C78", "#97BC62"])

    while len(colors) < node_count - 1:
        colors.append("#cccccc")
    colors.append("#4682B4")

    return colors


def _can_write_image() -> tuple[bool, str | None]:
    try:
        import kaleido  # noqa: F401
    except ImportError:
        return False, "kaleido not installed"

    try:
        plotly_version = Version(plotly.__version__)
    except InvalidVersion:
        return False, f"unrecognized plotly version '{plotly.__version__}'"

    if plotly_version < Version("6.1.1"):
        return False, f"plotly {plotly.__version__} < 6.1.1"

    return True, None


def main() -> None:
    nodes = build_nodes()
    aggregator_idx = len(nodes) - 1
    links = build_links(aggregator_idx)
    node_colors = build_colors(aggregator_idx, len(nodes))

    sankey = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors,
        ),
        link=dict(
            source=[link["source"] for link in links],
            target=[link["target"] for link in links],
            value=[link["value"] for link in links],
            color=[
                link.get("color", "rgba(135,206,235,0.3)") for link in links
            ],
        ),
        textfont=dict(size=12, family="Arial"),
        orientation="h",
    )

    fig = go.Figure(data=[sankey])
    fig.update_layout(
        title_text=(
            "<b>Data Flow of the Proposed Assurance Metric Framework</b><br>"
            "<span style='font-size:13px; font-weight:normal'>Hierarchical distribution "
            "from total metrics to theme categories and threshold sources.</span>"
        ),
        title_font_size=16,
        font_family="Arial",
        width=1200,
        height=700,
        margin=dict(l=20, r=20, t=80, b=20),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUTPUT_DIR / "metric_sankey.html"
    png_path = OUTPUT_DIR / "metric_sankey.png"
    fig.write_html(html_path, include_plotlyjs="cdn")
    can_write, reason = _can_write_image()
    if can_write:
        fig.write_image(png_path, scale=3)
        print("Saved Sankey diagram to", png_path)
    else:
        print(
            f"Skipping PNG export ({reason}). HTML output available at {html_path}. "
            "Install/upgrade plotly + kaleido to enable static export."
        )


if __name__ == "__main__":
    main()


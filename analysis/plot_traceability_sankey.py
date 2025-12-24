"""Build a traceability Sankey: Authorities -> Requirements -> Metrics -> Testbeds."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Callable

from packaging.version import InvalidVersion, Version
import plotly
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config" / "config_v2"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

# Order requested by the user for the left-most nodes.
AUTHORITY_ORDER = ["FAA", "EASA", "FDA", "STD", "FL", "XAI"]

# Color helpers for easy tweaking.
AUTHORITY_COLORS = {
    "FAA": "#1f77b4",
    "EASA": "#2ca02c",
    "FDA": "#d62728",
    "STD": "#9467bd",
    "FL": "#ff7f0e",
    "XAI": "#17becf",
}
TYPE_COLORS = {
    "authority": "#4f46e5",
    "requirement": "#10b981",
    "metric": "#f59e0b",
    "testbed": "#8b5cf6",
}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,a)'."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _parse_authorities(field: str | None) -> list[str]:
    """Split the authority cell on commas/semicolons and trim whitespace."""
    if not field:
        return []
    cleaned = field.replace(";", ",")
    return [part.strip() for part in cleaned.split(",") if part.strip()]


def _normalize_authority(code: str) -> str | None:
    """Return the base authority prefix (e.g., STD-001 -> STD)."""
    if not code:
        return None
    prefix = code.split("-")[0].strip().upper()
    return prefix if prefix else None


def _hash_color(label: str, alpha: float, s: int = 65, l: int = 52) -> str:
    """Generate a stable HSL-based rgba color for any label."""
    h = (abs(hash(label)) % 360)
    return f"hsla({h}, {s}%, {l}%, {alpha})"

def _load_requirement_titles() -> dict[str, str]:
    """Map requirement IDs to short labels using the requirements catalog."""
    titles: dict[str, str] = {}
    req_path = CONFIG_DIR / "B.Requirements.csv"
    with req_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            req_id = (row.get("req_ID") or row.get("Requirement ID") or "").strip()
            if not req_id:
                continue
            title = (row.get("Title") or "").strip()
            label = f"{req_id}: {title}" if title else req_id
            titles[req_id] = label
    return titles


def _load_requirement_authorities() -> dict[str, list[str]]:
    """Map requirement IDs to authority prefixes derived from normative clauses."""
    req_auths: dict[str, list[str]] = {}
    req_path = CONFIG_DIR / "B.Requirements.csv"
    with req_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            req_id = (row.get("req_ID") or row.get("Requirement ID") or "").strip()
            if not req_id:
                continue
            clauses = (row.get("Normative_Clauses_ID") or "").replace(";", ",")
            auths = []
            for part in clauses.split(","):
                code = part.strip()
                pref = _normalize_authority(code)
                if pref:
                    auths.append(pref)
            req_auths[req_id] = auths
    return req_auths


def _load_metrics_applicable() -> dict[str, dict[str, object]]:
    """Load metric metadata and applicable requirement IDs."""
    path = CONFIG_DIR / "C.1.1_Metrics_applicable.csv"
    metrics: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = (row.get("Metric ID") or "").strip()
            if not mid:
                continue
            name = (row.get("Metric Name") or "").strip()
            reqs_field = (row.get("Related Requirement ID(s)") or "").replace(";", ",")
            req_ids: list[str] = []
            for r in reqs_field.split(","):
                cleaned = r.strip().rstrip(".")
                if cleaned:
                    req_ids.append(cleaned)
            metrics[mid] = {"name": name, "req_ids": req_ids}
    return metrics


def _load_testbeds() -> list[dict[str, object]]:
    """Load testbed definitions and their metrics."""
    path = CONFIG_DIR / "E.Test_beds.csv"
    testbeds: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = (row.get("Test ID") or "").strip()
            if not tid:
                continue
            name = (row.get("Test name") or "").strip()
            metrics_field = (row.get("Metrics evaluated") or "").replace(";", ",")
            metrics = [m.strip() for m in metrics_field.split(",") if m.strip()]
            testbeds.append({"id": tid, "name": name, "metrics": metrics})
    return testbeds


def _add_node(label: str, node_type: str, nodes: list[str], types: list[str]) -> int:
    """Add a node if missing and return its index."""
    try:
        return nodes.index(label)
    except ValueError:
        nodes.append(label)
        types.append(node_type)
        return len(nodes) - 1


def _load_trace_links(
    req_labels: dict[str, str],
    metric_meta: dict[str, dict[str, object]],
    req_authorities: dict[str, list[str]],
    testbeds: list[dict[str, object]],
) -> tuple[list[str], list[str], Counter, list[str]]:
    """Build links using authoritative mappings from applicable metrics + test beds."""
    links: Counter[tuple[str, str]] = Counter()
    authorities_ordered: list[str] = [a for a in AUTHORITY_ORDER]
    extra_authorities: set[str] = set()
    requirements: set[str] = set()
    metrics: set[str] = set()
    testbed_labels: dict[str, str] = {}  # display -> hover full

    for tb in testbeds:
        tb_id = tb["id"]
        tb_name = tb["name"]
        tb_label = tb_id  # keep short label as requested
        testbed_labels[tb_label] = f"{tb_id}: {tb_name}"
        for metric_id in tb["metrics"]:
            if metric_id not in metric_meta:
                continue
            metric_name = metric_meta[metric_id]["name"]
            metric_label = f"{metric_id}: {metric_name}" if metric_name else metric_id
            metrics.add(metric_label)
            links[(metric_label, tb_label)] += 1

            for req_id in metric_meta[metric_id]["req_ids"]:
                req_label = req_labels.get(req_id, req_id)
                requirements.add(req_label)
                links[(req_label, metric_label)] += 1
                for auth in req_authorities.get(req_id, []):
                    if auth not in authorities_ordered and auth not in extra_authorities:
                        extra_authorities.add(auth)
                    links[(auth, req_label)] += 1

    ordered_nodes: list[str] = []
    ordered_types: list[str] = []

    for a in authorities_ordered:
        ordered_nodes.append(a)
        ordered_types.append("authority")
    for a in sorted(extra_authorities):
        if a not in authorities_ordered:
            ordered_nodes.append(a)
            ordered_types.append("authority")

    for req in sorted(requirements):
        ordered_nodes.append(req)
        ordered_types.append("requirement")
    for met in sorted(metrics):
        ordered_nodes.append(met)
        ordered_types.append("metric")
    for tb in sorted(testbed_labels):
        ordered_nodes.append(tb)
        ordered_types.append("testbed")

    hover_labels: list[str] = []
    for lbl in ordered_nodes:
        if lbl in testbed_labels:
            hover_labels.append(testbed_labels[lbl])
        else:
            hover_labels.append(lbl)

    return ordered_nodes, ordered_types, links, hover_labels


def _build_colors(
    nodes: list[str], node_types: list[str]
) -> tuple[list[str], dict[str, int], Callable[[str], str]]:
    """Create node colors and a lookup to infer link colors by source node."""
    node_colors: list[str] = []
    for label, ntype in zip(nodes, node_types):
        if ntype == "authority":
            node_colors.append(AUTHORITY_COLORS.get(label, TYPE_COLORS["authority"]))
        elif ntype == "requirement":
            node_colors.append(_hash_color(label, alpha=0.9))
        elif ntype == "metric":
            node_colors.append(_hash_color(label, alpha=0.9, s=70, l=60))
        else:  # testbed or other
            node_colors.append(TYPE_COLORS.get(ntype, "#cccccc"))

    node_index = {label: idx for idx, label in enumerate(nodes)}

    def link_color(src_label: str) -> str:
        src_type = node_types[node_index[src_label]]
        if src_type == "authority":
            base = AUTHORITY_COLORS.get(src_label, TYPE_COLORS["authority"])
            return _hex_to_rgba(base, 0.45)
        if src_type == "requirement":
            return _hash_color(src_label, alpha=0.35)
        if src_type == "metric":
            return _hash_color(src_label, alpha=0.35, s=70, l=60)
        return _hex_to_rgba(TYPE_COLORS.get(src_type, "#999999"), 0.25)

    return node_colors, node_index, link_color


def _write_mapping_file(
    metric_meta: dict[str, dict[str, object]],
    req_labels: dict[str, str],
    req_authorities: dict[str, list[str]],
    testbeds: list[dict[str, object]],
) -> None:
    """Regenerate F.Mapping.csv from authoritative metric + testbed sources."""
    out_path = CONFIG_DIR / "F.Mapping.csv"
    rows: list[list[str]] = []
    header = [
        "Test ID",
        "Test Name",
        "Metric ID",
        "Metric Name",
        "Requirement ID",
        "Authorities Source",
    ]
    rows.append(header)

    for tb in testbeds:
        tid = tb["id"]
        tname = tb["name"]
        for mid in tb["metrics"]:
            if mid not in metric_meta:
                continue
            mname = metric_meta[mid]["name"]
            req_ids = metric_meta[mid]["req_ids"]
            for rid in req_ids:
                auths = req_authorities.get(rid, [])
                auth_field = ", ".join(dict.fromkeys(auths))  # preserve order, unique
                rows.append([tid, tname, mid, mname, rid, auth_field])

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _can_write_image() -> tuple[bool, str | None]:
    """Return (can_write, reason_if_not)."""
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


def build_figure(nodes: list[str], node_types: list[str], links: Counter, hover_labels: list[str]) -> go.Figure:
    """Assemble the Sankey figure."""
    node_colors, node_index, link_color_fn = _build_colors(nodes, node_types)

    x_positions = {
        "authority": 0.0,
        "requirement": 0.32,
        "metric": 0.65,
        "testbed": 0.98,
    }
    x_coords = [x_positions.get(ntype, 0.5) for ntype in node_types]

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []

    for (src_label, tgt_label), count in links.items():
        if src_label not in node_index or tgt_label not in node_index:
            continue
        sources.append(node_index[src_label])
        targets.append(node_index[tgt_label])
        values.append(count)
        link_colors.append(link_color_fn(src_label))

    sankey = go.Sankey(
        node=dict(
            pad=16,
            thickness=18,
            line=dict(color="rgba(0,0,0,0.2)", width=0.5),
            label=nodes,
            color=node_colors,
            x=x_coords,
            customdata=hover_labels,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
        orientation="h",
    )

    fig = go.Figure(data=[sankey])
    fig.update_layout(
        title_text=(
            "<b>Traceability Flow</b><br>"
            "<span style='font-size:13px; font-weight:normal'>"
            "Authorities → Requirements → Applicable Metrics → Test Beds"
            "</span>"
        ),
        title_font_size=16,
        font_family="Arial",
        width=1400,
        height=800,
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


def main() -> None:
    req_labels = _load_requirement_titles()
    metric_meta = _load_metrics_applicable()
    req_authorities = _load_requirement_authorities()
    testbeds = _load_testbeds()

    nodes, node_types, links, hover_labels = _load_trace_links(
        req_labels, metric_meta, req_authorities, testbeds
    )

    _write_mapping_file(metric_meta, req_labels, req_authorities, testbeds)

    fig = build_figure(nodes, node_types, links, hover_labels)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUTPUT_DIR / "traceability_sankey.html"
    png_path = OUTPUT_DIR / "traceability_sankey.png"
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








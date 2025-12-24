"""
GSN builder for FL-PdM assurance case derived from Concentric Assurance Map (CAM).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from textwrap import fill

import matplotlib.pyplot as plt
from graphviz import Digraph
from graphviz.backend import ExecutableNotFound
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, RegularPolygon

GOAL = "Goal"
STRATEGY = "Strategy"
CONTEXT = "Context"
JUSTIFICATION = "Justification"
SOLUTION = "Solution"


PILLAR_GOALS: Dict[str, Dict[str, str]] = {
    "G1": {
        "label": "Federated Data Assurance & Governance",
        "color": "#4e79a7",
        "domains": {"Data & Governance"},
    },
    "G2": {
        "label": "Federated Learning Process Assurance",
        "color": "#f28e2c",
        "domains": {"Lifecycle & System Reliability", "Communication & Efficiency"},
    },
    "G3": {
        "label": "Model Robustness & Security",
        "color": "#e15759",
        "domains": {"Model Performance & Robustness", "Privacy & Security"},
    },
    "G4": {
        "label": "System Trustworthiness & Explainability",
        "color": "#59a14f",
        "domains": {"Fairness & Equity"},
    },
}


STATUS_COLORS = {
    "PASS": "#6ba368",
    "FAIL": "#d45050",
    "ALARM": "#9b59b6",
    "MONITOR": "#f2c14e",
    "": "#bab0ac",
}


def goal_fill_color(node: Node) -> Tuple[str, str]:
    domain = node.metadata.get("domain")
    if isinstance(domain, str):
        for info in PILLAR_GOALS.values():
            if domain in info["domains"]:
                return info["color"], "#ffffff"
    pillar = node.metadata.get("pillar")
    if isinstance(pillar, str):
        info = PILLAR_GOALS.get(pillar)
        if info:
            return info["color"], "#ffffff"
    return "#ffffff", "#000000"


@dataclass
class Node:
    id: str
    type: str
    label: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    relation: str = "supports"


def load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_cam(cam_path: Path) -> Mapping[str, object]:
    data = load_json(cam_path)
    required = ["requirements", "metrics", "tests", "links", "results_overlay"]
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"CAM JSON is missing required keys: {', '.join(missing)}")
    return data


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalise_thresholds(metric: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    thresholds = metric.get("thresholds", {}) or {}
    return {
        level: {
            "value": entry.get("value"),
            "op": entry.get("op"),
            "assumptions": entry.get("assumptions") or {},
        }
        for level, entry in thresholds.items()
    }


def format_threshold_summary(thresholds: Mapping[str, Mapping[str, object]]) -> str:
    parts = []
    for level in ("Catastrophic", "Hazardous", "Major"):
        entry = thresholds.get(level, {})
        if entry:
            val = entry.get("value")
            op = entry.get("op", "")
            if val is None or (isinstance(val, float) and math.isnan(val)):
                parts.append(f"{level[:3]}: n/a")
            else:
                parts.append(f"{level[:3]}: {op} {val}")
    return "; ".join(parts)


def build_gsn(
    cam_data: Mapping[str, object],
    out_dir: Path,
) -> Tuple[List[Node], List[Edge], Mapping[str, object]]:
    requirements: Sequence[Mapping[str, object]] = cam_data["requirements"]  # type: ignore[assignment]
    metrics: Sequence[Mapping[str, object]] = cam_data["metrics"]  # type: ignore[assignment]
    tests: Sequence[Mapping[str, object]] = cam_data["tests"]  # type: ignore[assignment]
    overlays: Sequence[Mapping[str, object]] = cam_data["results_overlay"]  # type: ignore[assignment]
    links: Mapping[str, Sequence[Mapping[str, str]]] = cam_data["links"]  # type: ignore[assignment]

    req_to_metrics: Dict[str, List[str]] = {}
    for link in links.get("requirement_to_metric", []):
        req_to_metrics.setdefault(link["requirement_id"], []).append(link["metric_id"])

    metric_to_tests: Dict[str, List[str]] = {}
    for link in links.get("metric_to_test", []):
        metric_to_tests.setdefault(link["metric_id"], []).append(link["test_id"])

    overlay_map: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    for entry in overlays:
        key = (entry["test_id"], entry["metric_id"])
        overlay_map.setdefault(key, []).append(entry)

    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []

    def add_node(node: Node) -> None:
        if node.id not in nodes:
            nodes[node.id] = node

    def add_edge(source: str, target: str, relation: str = "supports") -> None:
        edges.append(Edge(source, target, relation))

    # Top claim and primary structure
    add_node(Node("G0", GOAL, "FL-PdM meets assurance objectives within intended ODD"))
    add_node(Node("S1", STRATEGY, "Decompose by assurance pillars"))
    add_edge("G0", "S1", "strategy")

    add_node(Node("C_ODD", CONTEXT, "Operational Design Domain scope"))
    add_edge("G0", "C_ODD", "context")

    add_node(Node("J_FHA", JUSTIFICATION, "FHA-based thresholds for safety causal metrics"))
    add_edge("G0", "J_FHA", "justification")

    add_node(Node("J_Bounds", JUSTIFICATION, "Literature-bounded caps for supporting metrics"))
    add_edge("G0", "J_Bounds", "justification")

    pillar_lookup: Dict[str, str] = {}
    for goal_id, info in PILLAR_GOALS.items():
        add_node(Node(goal_id, GOAL, info["label"], {"pillar": goal_id, "color": info["color"]}))
        add_edge("S1", goal_id, "supports")
        for domain in info["domains"]:
            pillar_lookup[domain] = goal_id

    # Requirement goals
    for requirement in sorted(requirements, key=lambda r: r["id"]):
        req_id = requirement["id"]
        goal_id = f"G_R_{req_id}"
        label = f"{req_id}: {requirement.get('title', '').strip()}"
        metadata = {
            "domain": requirement.get("domain"),
            "source": requirement.get("source_authority_id"),
        }
        add_node(Node(goal_id, GOAL, label, metadata))
        domain = requirement.get("domain")
        parent_goal = pillar_lookup.get(domain, "G4")
        add_edge(parent_goal, goal_id, "supports")

    # Requirement -> Metric strategies and goals
    for requirement in sorted(requirements, key=lambda r: r["id"]):
        req_id = requirement["id"]
        req_goal_id = f"G_R_{req_id}"
        metric_ids = sorted(req_to_metrics.get(req_id, []))
        if not metric_ids:
            continue
        strategy_id = f"S_R2M_{req_id}"
        add_node(Node(strategy_id, STRATEGY, f"Operationalise {req_id} via metrics"))
        add_edge(req_goal_id, strategy_id, "strategy")
        for metric_id in metric_ids:
            metric_goal = f"G_M_{metric_id}"
            add_edge(strategy_id, metric_goal, "supports")

    # Metric goals
    for metric in sorted(metrics, key=lambda m: m["id"]):
        metric_id = metric["id"]
        goal_id = f"G_M_{metric_id}"
        label = f"{metric_id}: {metric.get('name', '').strip()}"
        thresholds = normalise_thresholds(metric)
        metadata = {
            "unit": metric.get("unit"),
            "direction": metric.get("direction"),
            "hazard_link": metric.get("hazard_link"),
            "threshold_summary": format_threshold_summary(thresholds),
        }
        add_node(Node(goal_id, GOAL, label, metadata))
        if metric.get("gated"):
            assumptions = thresholds.get("Catastrophic", {}).get("assumptions", {})
            if assumptions:
                context_id = f"C_FHA_{metric_id}"
                context_label = (
                    f"FHA assumptions {metric_id}: "
                    + ", ".join(f"{k}={v}" for k, v in assumptions.items())
                )
                add_node(Node(context_id, CONTEXT, context_label, {"metric": metric_id}))
                add_edge(goal_id, context_id, "context")
        else:
            add_edge(goal_id, "J_Bounds", "justification")

    # Test contexts
    test_map = {exp["id"]: exp for exp in tests}
    for test in sorted(tests, key=lambda e: e["id"]):
        exp_id = test["id"]
        params = test.get("params") or {}
        if params:
            details = ", ".join(f"{k}={v}" for k, v in params.items())
        else:
            details = "No additional parameters"
        context_id = f"C_SCEN_{exp_id}"
        add_node(Node(context_id, CONTEXT, f"{exp_id} scenario: {details}", {"test": exp_id}))

    # Metric -> Evidence strategies
    for metric in sorted(metrics, key=lambda m: m["id"]):
        metric_id = metric["id"]
        goal_id = f"G_M_{metric_id}"
        linked_tests = sorted(set(metric_to_tests.get(metric_id, [])))
        overlay_pairs: List[Tuple[str, Mapping[str, object]]] = []
        for exp_id in linked_tests:
            evidences = overlay_map.get((exp_id, metric_id), [])
            for idx, evidence in enumerate(evidences):
                overlay_pairs.append((f"{exp_id}_{idx}", evidence))
        if overlay_pairs:
            strategy_id = f"S_M2E_{metric_id}"
            add_node(Node(strategy_id, STRATEGY, f"Assess {metric_id} via test evidence"))
            add_edge(goal_id, strategy_id, "strategy")
            for exp_id in linked_tests:
                context_id = f"C_SCEN_{exp_id}"
                if context_id in nodes:
                    add_edge(strategy_id, context_id, "context")
            for suffix, evidence in overlay_pairs:
                evidence_id = f"Sn_{suffix}_{metric_id}"
                status = str(evidence.get("status", "")).upper()
                pass_level = evidence.get("pass_level", "None")
                observed = evidence.get("observed")
                unit = evidence.get("unit", "")
                label = (
                    f"{evidence['test_id']}×{metric_id}: {status} "
                    f"@ {observed} {unit} (level={pass_level})"
                )
                evidence_metadata = {
                    "test_id": evidence["test_id"],
                    "metric_id": metric_id,
                    "observed": observed,
                    "unit": unit,
                    "direction": evidence.get("direction"),
                    "thresholds": evidence.get("thresholds"),
                    "pass_level": pass_level,
                    "status": status,
                    "comment": evidence.get("comment"),
                    "ci95": evidence.get("ci95"),
                }
                add_node(Node(evidence_id, SOLUTION, label, evidence_metadata))
                add_edge(strategy_id, evidence_id, "solution")
                context_id = f"C_SCEN_{evidence['test_id']}"
                if context_id in nodes:
                    add_edge(evidence_id, context_id, "context")
        else:
            # No evidence: create placeholder node
            placeholder_id = f"Sn_missing_{metric_id}"
            add_node(Node(placeholder_id, SOLUTION, f"No evidence linked for {metric_id}", {"metric": metric_id}))
            add_edge(goal_id, placeholder_id, "solution")

    # Validation summary
    metrics_with_evidence = {
        node.metadata.get("metric_id")
        for node in nodes.values()
        if node.type == SOLUTION and node.metadata.get("metric_id")
    }

    missing_metric_evidence = [
        metric["id"]
        for metric in sorted(metrics, key=lambda m: m["id"])
        if metric["id"] not in metrics_with_evidence
    ]

    missing_requirement_links = [
        requirement["id"]
        for requirement in sorted(requirements, key=lambda r: r["id"])
        if requirement["id"] not in req_to_metrics
    ]

    validation = {
        "requirement_count": len(requirements),
        "metric_count": len(metrics),
        "test_count": len(tests),
        "evidence_count": sum(
            1
            for node in nodes.values()
            if node.type == SOLUTION and node.metadata.get("metric_id")
        ),
        "missing_requirement_links": missing_requirement_links,
        "missing_metric_evidence": missing_metric_evidence,
    }

    return list(nodes.values()), edges, validation


def write_json(path: Path, content: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(content, fp, indent=2, ensure_ascii=False)


def render_graph(nodes: Sequence[Node], edges: Sequence[Edge], output_path: Path) -> None:
    graph = Digraph("GSN", graph_attr={"rankdir": "TB", "fontsize": "10"})
    graph.attr("node", fontname="Helvetica", style="filled", fontsize="10")

    for node in sorted(nodes, key=lambda n: n.id):
        attrs = {"label": node.label}
        if node.type == GOAL:
            fill, font_color = goal_fill_color(node)
            attrs.update({"shape": "rect", "style": "rounded,filled", "fillcolor": fill, "fontcolor": font_color})
        elif node.type == STRATEGY:
            attrs.update({"shape": "parallelogram", "fillcolor": "#d0ece7"})
        elif node.type == CONTEXT:
            attrs.update({"shape": "ellipse", "fillcolor": "#f0f0f0"})
        elif node.type == JUSTIFICATION:
            attrs.update({"shape": "hexagon", "fillcolor": "#e8daef"})
        elif node.type == SOLUTION:
            status = str(node.metadata.get("status", "")).upper()
            fill = STATUS_COLORS.get(status, "#bab0ac")
            attrs.update({"shape": "note", "fillcolor": fill})
        graph.node(node.id, **attrs)

    for edge in edges:
        attrs = {}
        if edge.relation == "strategy":
            attrs["style"] = "dashed"
        elif edge.relation == "context":
            attrs["style"] = "dotted"
        elif edge.relation == "justification":
            attrs["style"] = "dotted"
        graph.edge(edge.source, edge.target, **attrs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path = output_path.with_suffix(".dot")
    graph.save(filename=str(dot_path))
    try:
        graph.render(output_path.with_suffix(""), format="svg", cleanup=True)
    except ExecutableNotFound:
        render_graph_matplotlib(nodes, edges, output_path.with_suffix(".svg"))


def render_graph_matplotlib(nodes: Sequence[Node], edges: Sequence[Edge], svg_path: Path) -> None:
    node_map = {node.id: node for node in nodes}
    incoming: Dict[str, List[Edge]] = {node.id: [] for node in nodes}
    for edge in edges:
        incoming.setdefault(edge.target, []).append(edge)

    depth_cache: Dict[str, float] = {}

    def depth(node_id: str) -> float:
        if node_id in depth_cache:
            return depth_cache[node_id]
        parents = incoming.get(node_id, [])
        if not parents:
            depth_cache[node_id] = 0.0
            return 0.0
        parent_depths = []
        for edge in parents:
            increment = 0.0 if edge.relation in {"context", "justification"} else 1.0
            parent_depths.append(depth(edge.source) + increment)
        depth_cache[node_id] = max(parent_depths)
        return depth_cache[node_id]

    for node_id in node_map:
        depth(node_id)

    levels: Dict[float, List[str]] = {}
    for node_id, value in depth_cache.items():
        levels.setdefault(value, []).append(node_id)

    sorted_levels = sorted(levels.keys())
    max_depth = max(sorted_levels) if sorted_levels else 1.0

    positions: Dict[str, Tuple[float, float]] = {}
    for lvl in sorted_levels:
        nodes_at_level = sorted(levels[lvl])
        count = len(nodes_at_level)
        for idx, node_id in enumerate(nodes_at_level):
            x = (idx + 1) / (count + 1)
            y = 1.0 - (lvl / (max_depth + 1.0))
            positions[node_id] = (x, y)

    fig, ax = plt.subplots(figsize=(22, 14))
    ax.axis("off")

    def draw_node(node: Node, x: float, y: float) -> None:
        width = 0.18
        height = 0.08
        wrapped = fill(node.label, width=28)
        if node.type == GOAL:
            fill_color, font_color = goal_fill_color(node)
            patch = FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.02",
                facecolor=fill_color,
                edgecolor="#333333",
            )
            ax.add_patch(patch)
            ax.text(x, y, wrapped, ha="center", va="center", fontsize=8, color=font_color)
        elif node.type == STRATEGY:
            patch = FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0",
                facecolor="#d0ece7",
                edgecolor="#333333",
            )
            ax.add_patch(patch)
            ax.text(x, y, wrapped, ha="center", va="center", fontsize=8)
        elif node.type == CONTEXT:
            patch = Ellipse((x, y), width, height, facecolor="#f0f0f0", edgecolor="#333333")
            ax.add_patch(patch)
            ax.text(x, y, wrapped, ha="center", va="center", fontsize=8)
        elif node.type == JUSTIFICATION:
            patch = RegularPolygon((x, y), numVertices=6, radius=height / 1.2, facecolor="#e8daef", edgecolor="#333333")
            ax.add_patch(patch)
            ax.text(x, y, wrapped, ha="center", va="center", fontsize=8)
        elif node.type == SOLUTION:
            status = str(node.metadata.get("status", "")).upper()
            fill_color = STATUS_COLORS.get(status, "#bab0ac")
            patch = FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0",
                facecolor=fill_color,
                edgecolor="#333333",
            )
            ax.add_patch(patch)
            ax.text(x, y, wrapped, ha="center", va="center", fontsize=7)

    for node_id, (x, y) in positions.items():
        draw_node(node_map[node_id], x, y)

    for edge in edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        style = "solid"
        if edge.relation == "context":
            style = "dashed"
        elif edge.relation == "justification":
            style = "dotted"
        arrow = FancyArrowPatch(
            (x1, y1 - 0.02),
            (x2, y2 + 0.02),
            arrowstyle="->",
            linewidth=1.0,
            linestyle=style,
            color="#555555",
            connectionstyle="arc3,rad=0.1",
        )
        ax.add_patch(arrow)

    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    path: Path,
    nodes: Sequence[Node],
    validation: Mapping[str, object],
) -> None:
    goal_nodes = [node for node in nodes if node.type == GOAL and node.id.startswith("G_R_")]
    metric_nodes = [node for node in nodes if node.type == GOAL and node.id.startswith("G_M_")]
    solution_nodes = [node for node in nodes if node.type == SOLUTION and not node.id.startswith("Sn_missing")]

    lines = [
        "# GSN Assurance Case",
        "",
        "This report summarises the Goal Structuring Notation (GSN) assurance case generated from the Concentric Assurance Map (CAM) artefacts.",
        "",
        "## Overview",
        "",
        "- Top claim: FL-PdM system meets safety and assurance objectives within the intended ODD.",
        "- Pillar coverage: 4 primary assurance goals reflecting data governance, process assurance, robustness & security, and trustworthiness & explainability.",
        f"- Requirement goals: {len(goal_nodes)}",
        f"- Metric goals: {len(metric_nodes)}",
        f"- Evidence nodes: {len(solution_nodes)}",
        "",
        "![GSN Assurance Case](GSN.svg)",
        "",
        "## Coverage Summary",
        "",
        f"- Requirements with linked metrics: {len(goal_nodes)} / {validation.get('requirement_count')}",
        f"- Metrics with evidence: {len(solution_nodes)} / {validation.get('metric_count')}",
        "",
        "## Evidence Table",
        "",
        "| Evidence ID | Test | Metric | Status | Pass Level | Observed | Unit | Comment |",
        "|-------------|------------|--------|--------|------------|----------|------|---------|",
    ]

    for node in sorted(solution_nodes, key=lambda n: n.id):
        metadata = node.metadata
        evidence_id = node.id
        test = metadata.get("test_id", "")
        metric = metadata.get("metric_id", "")
        status = metadata.get("status", "")
        pass_level = metadata.get("pass_level", "")
        observed = metadata.get("observed", "")
        unit = metadata.get("unit", "")
        comment = metadata.get("comment", "") or ""
        comment = str(comment).replace("|", "\\|")
        lines.append(
            f"| {evidence_id} | {test} | {metric} | {status} | {pass_level} | {observed} | {unit} | {comment} |"
        )

    with path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def write_validation_report(path: Path, validation: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(validation, fp, indent=2)


def build_cli(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate GSN assurance case from CAM artefacts.")
    parser.add_argument("--cam", type=Path, required=True, help="Path to cam.json")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for GSN artefacts")
    args = parser.parse_args(argv)

    cam_data = load_cam(args.cam)
    ensure_output_dir(args.out_dir)

    nodes, edges, validation = build_gsn(cam_data, args.out_dir)

    gsn_json = {
        "nodes": [
            {
                "id": node.id,
                "type": node.type,
                "label": node.label,
                "metadata": node.metadata,
            }
            for node in sorted(nodes, key=lambda n: n.id)
        ],
        "edges": [
            {"from": edge.source, "to": edge.target, "relation": edge.relation}
            for edge in edges
        ],
    }
    write_json(args.out_dir / "GSN.json", gsn_json)
    render_graph(nodes, edges, args.out_dir / "GSN")
    write_markdown(args.out_dir / "GSN.md", nodes, validation)
    write_validation_report(args.out_dir / "validation.json", validation)


def main() -> None:
    build_cli()


if __name__ == "__main__":
    main()



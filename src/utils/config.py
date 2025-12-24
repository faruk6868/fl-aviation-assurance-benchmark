"""Configuration loading utilities for metric metadata and thresholds."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import pandas as pd

LEVEL_LABELS: Sequence[Tuple[str, str]] = (
    ("Catastrophic", "Cat"),
    ("Hazardous", "Haz"),
    ("Major", "Maj"),
)

# ---------------------------------------------------------------------------
# Legacy dataclasses (v1 configuration)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricMetadata:
    """Structured representation of a metric definition (legacy v1)."""

    id: str
    name: str
    direction: str
    unit: str
    method: str
    hazard_link: str
    notes: str | None = None


@dataclass(frozen=True)
class RequirementMapping:
    """Associates a metric with regulatory requirements and category (legacy v1)."""

    category: str | None = None
    requirements: str | None = None
    optimal_target: str | None = None


# ---------------------------------------------------------------------------
# New dataclasses (v2 configuration)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormativeClause:
    """Describes a normative clause and its provenance."""

    clause_id: str
    text: str | None
    source_document: str | None
    authority: str | None
    referenced_requirements: str | None


@dataclass(frozen=True)
class RequirementRecord:
    """Structured representation of a single requirement statement."""

    req_id: str
    title: str | None
    statement: str | None
    section: str | None = None
    subsection: str | None = None
    normative_clauses: Sequence[str] = ()


@dataclass(frozen=True)
class MetricDefinition:
    """Detailed definition for a metric in the new catalog."""

    metric_id: str
    name: str | None
    definition: str | None
    justification: str | None
    related_requirements: Sequence[str] = ()


@dataclass(frozen=True)
class ThresholdRange:
    """Holds min/max target values and provenance for a metric."""

    metric_id: str
    metric_name: str | None
    min_text: str | None
    max_text: str | None
    min_value: float | None
    max_value: float | None
    unit: str | None
    elicitation_method: str | None
    source_category: str | None
    primary_source: str | None
    enforce_threshold: bool = True


@dataclass(frozen=True)
class TestBedDefinition:
    """Definition of an assurance test bed and associated metrics."""

    test_id: str
    name: str | None
    definition: str | None
    justification: str | None
    metrics: Sequence[str]


@dataclass(frozen=True)
class MappingEntry:
    """Mapping between tests, metrics, requirements and authorities."""

    test_id: str
    test_name: str | None
    metric_id: str
    metric_name: str | None
    requirement_id: str | None
    authority_source: str | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with BOM/encoding fallbacks and trimmed headers."""

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _clean_str(value: object) -> str | None:
    """Convert arbitrary cell values to trimmed strings."""

    if isinstance(value, str):
        text = value.strip()
        return text or None
    if value is None:
        return None
    if isinstance(value, (int, float)) and (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _split_multi_value_field(value: object) -> List[str]:
    """Split semi-structured requirement/reference lists into clean tokens."""

    raw = _clean_str(value)
    if not raw:
        return []
    tokens = re.split(r"[;,]", raw)
    return [token.strip() for token in tokens if token.strip()]


def _attempt_float(value: str | None) -> float | None:
    """Try to coerce a textual threshold bound into a float."""

    if not value:
        return None
    cleaned = re.sub(r"[^\d.+-eE]", "", value)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_metric_list(field: object) -> List[str]:
    """Extract metric identifiers from bullet/semicolon separated strings."""

    raw = _clean_str(field)
    if not raw:
        return []
    metrics: List[str] = []
    for chunk in re.split(r"[;\n]", raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        token = chunk.split()[0].strip(",")
        metrics.append(token)
    return metrics


# ---------------------------------------------------------------------------
# Legacy loaders (v1) – retained for backward compatibility
# ---------------------------------------------------------------------------


def load_metric_metadata(path: Path) -> Dict[str, MetricMetadata]:
    """Load metric definitions from ``metric_threshold_metadata.json`` (legacy)."""

    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    metrics: Dict[str, MetricMetadata] = {}
    for entry in payload["metrics"]:
        metrics[entry["id"]] = MetricMetadata(
            id=entry["id"],
            name=entry["name"],
            direction=entry["direction"],
            unit=entry["unit"],
            method=entry["method"],
            hazard_link=entry["hazard_link"],
            notes=entry.get("notes"),
        )

    return metrics


def load_threshold_table(path: Path) -> pd.DataFrame:
    """Load a single threshold CSV as a dataframe indexed by metric id (legacy)."""

    df = _read_csv(path)
    if "Metric_ID" not in df.columns:
        raise ValueError(f"Threshold file {path} missing Metric_ID column")
    return df.set_index("Metric_ID")


def load_all_thresholds(config_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load Catastrophic, Hazardous, and Major threshold tables (legacy)."""

    thresholds: Dict[str, pd.DataFrame] = {}
    for level in ("Catastrophic", "Hazardous", "Major"):
        file_name = f"thresholds_{level}.csv"
        csv_path = config_dir / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing threshold file: {csv_path}")
        thresholds[level] = load_threshold_table(csv_path)
    return thresholds


def extract_thresholds(metric_id: str, thresholds: Mapping[str, pd.DataFrame]) -> Tuple[float, float, float]:
    """Return (Cat, Haz, Maj) threshold values for a metric (legacy helper)."""

    values: List[float] = []
    for level in ("Catastrophic", "Hazardous", "Major"):
        df = thresholds[level]
        values.append(df.loc[metric_id, "Threshold_Value"])
    return tuple(values)  # type: ignore[return-value]


def load_requirement_mapping(path: Path) -> Dict[str, RequirementMapping]:
    """Load mapping between metrics and regulatory requirements (legacy)."""

    df = _read_csv(path)
    required_columns = {"Metric ID", "Category", "Related Requirements"}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(sorted(required_columns - set(df.columns)))
        raise ValueError(f"Requirement mapping file missing columns: {missing}")

    mapping: Dict[str, RequirementMapping] = {}
    for _, row in df.iterrows():
        metric_id = str(row.get("Metric ID", "")).strip()
        if not metric_id:
            continue
        category = str(row.get("Category", "")).strip() or None
        requirements = str(row.get("Related Requirements", "")).strip() or None
        optimal_target = str(row.get("Optimal Target", "")).strip() or None
        mapping[metric_id] = RequirementMapping(
            category=category,
            requirements=requirements,
            optimal_target=optimal_target,
        )
    return mapping


# ---------------------------------------------------------------------------
# New loaders (v2 configuration)
# ---------------------------------------------------------------------------


def load_normative_clauses(path: Path) -> Dict[str, NormativeClause]:
    """Parse the normative clauses table."""

    df = _read_csv(path)
    clauses: Dict[str, NormativeClause] = {}
    for _, row in df.iterrows():
        clause_id = _clean_str(row.get("Normative Clause  ID")) or _clean_str(row.get("ID"))
        if not clause_id:
            continue
        clauses[clause_id] = NormativeClause(
            clause_id=clause_id,
            text=_clean_str(row.get("Normative Clauses")),
            source_document=_clean_str(row.get("Source Document")),
            authority=_clean_str(row.get("Authorities / Source")),
            referenced_requirements=_clean_str(row.get("ID.s")),
        )
    return clauses


def load_requirements_index(path: Path) -> Dict[str, RequirementRecord]:
    """Load the requirements catalog (section/subsection aware)."""

    df = _read_csv(path)
    records: Dict[str, RequirementRecord] = {}
    for _, row in df.iterrows():
        req_id = _clean_str(row.get("req_ID"))
        if not req_id:
            continue
        records[req_id] = RequirementRecord(
            req_id=req_id,
            title=_clean_str(row.get("Title")),
            statement=_clean_str(row.get("Requirement_Statement")),
            section=_clean_str(row.get("Section")),
            subsection=_clean_str(row.get("Subsection")),
            normative_clauses=_split_multi_value_field(row.get("Normative_Clauses_ID")),
        )
    return records


def load_metric_catalog_v2(path: Path) -> Dict[str, MetricDefinition]:
    """Load the metric catalog introduced in config_v2."""

    df = _read_csv(path)
    catalog: Dict[str, MetricDefinition] = {}
    for _, row in df.iterrows():
        metric_id = _clean_str(row.get("Metric ID"))
        if not metric_id:
            continue
        catalog[metric_id] = MetricDefinition(
            metric_id=metric_id,
            name=_clean_str(row.get("Metric Name")),
            definition=_clean_str(row.get("Metric Definition")),
            justification=_clean_str(row.get("Justification")) or _clean_str(row.get("Justification (why selected)")),
            related_requirements=_split_multi_value_field(row.get("Related Requirement ID(s)")),
        )
    return catalog


def load_metric_applicability(
    applicable_path: Path, excluded_path: Path
) -> Tuple[Dict[str, MetricDefinition], Dict[str, MetricDefinition]]:
    """Load applicable and excluded metric tables as dictionaries keyed by ID."""

    applicable = load_metric_catalog_v2(applicable_path)
    excluded = load_metric_catalog_v2(excluded_path)
    return applicable, excluded


def load_threshold_ranges(path: Path) -> Dict[str, ThresholdRange]:
    """Load aggregated min/max threshold information for each metric."""

    df = _read_csv(path)
    ranges: Dict[str, ThresholdRange] = {}
    for _, row in df.iterrows():
        metric_id = _clean_str(row.get("Metric ID"))
        if not metric_id:
            continue
        min_text = _clean_str(row.get("Threshold Min"))
        max_text = _clean_str(row.get("Threshold max"))
        enforce = True
        min_value = _attempt_float(min_text)
        max_value = _attempt_float(max_text)
        if min_value is None and max_value is None:
            enforce = False
        ranges[metric_id] = ThresholdRange(
            metric_id=metric_id,
            metric_name=_clean_str(row.get("Metric Name")),
            min_text=min_text,
            max_text=max_text,
            min_value=min_value,
            max_value=max_value,
            unit=_clean_str(row.get("Unit")),
            elicitation_method=_clean_str(row.get("Threshold Eliciation Method")),
            source_category=_clean_str(row.get("Source Category")),
            primary_source=_clean_str(row.get("Primary Source")),
            enforce_threshold=enforce,
        )
    return ranges


def load_test_beds(path: Path) -> Dict[str, TestBedDefinition]:
    """Load assurance test bed definitions and associated metrics."""

    df = _read_csv(path)
    beds: Dict[str, TestBedDefinition] = {}
    for _, row in df.iterrows():
        test_id = _clean_str(row.get("Test ID"))
        if not test_id:
            continue
        beds[test_id.upper()] = TestBedDefinition(
            test_id=test_id.upper(),
            name=_clean_str(row.get("Test name")),
            definition=_clean_str(row.get("Test definition")),
            justification=_clean_str(row.get("Justification")),
            metrics=_parse_metric_list(row.get("Metrics evaluated")),
        )
    return beds


def load_mapping_entries(path: Path) -> List[MappingEntry]:
    """Load the combined requirement ↔ metric ↔ test mapping."""

    df = _read_csv(path)
    entries: List[MappingEntry] = []
    for _, row in df.iterrows():
        metric_id = _clean_str(row.get("Metric ID"))
        test_id = _clean_str(row.get("Test ID"))
        if not metric_id or not test_id:
            continue
        entries.append(
            MappingEntry(
                test_id=test_id,
                test_name=_clean_str(row.get("Test Name")),
                metric_id=metric_id,
                metric_name=_clean_str(row.get("Metric Name")),
                requirement_id=_clean_str(row.get("Requirement ID")),
                authority_source=_clean_str(row.get("Authorities Source")),
            )
        )
    return entries


def _infer_direction(threshold_range: ThresholdRange | None) -> str:
    """Guess whether higher or lower values are better based on threshold ordering."""

    if threshold_range is None:
        return "max"
    low = threshold_range.min_value
    high = threshold_range.max_value
    if low is None and high is None:
        return "max"
    if low is None:
        return "min" if high is not None and high <= 0 else "max"
    if high is None:
        return "max"
    return "max" if low <= high else "min"


def _derive_threshold_levels(range_obj: ThresholdRange, direction: str) -> Dict[str, float | None]:
    """Approximate Cat/Haz/Maj thresholds from aggregated min/max."""

    return {"min": range_obj.min_value, "max": range_obj.max_value}


def build_metric_metadata_from_v2(
    catalog: Mapping[str, MetricDefinition],
    thresholds: Mapping[str, ThresholdRange],
    mappings: Sequence[MappingEntry],
) -> Dict[str, MetricMetadata]:
    """Create :class:`MetricMetadata` entries from the v2 catalog."""

    authority_map: Dict[str, str] = {}
    for entry in mappings:
        if entry.metric_id not in authority_map and entry.authority_source:
            authority_map[entry.metric_id] = entry.authority_source

    metadata: Dict[str, MetricMetadata] = {}
    for metric_id, definition in catalog.items():
        threshold_range = thresholds.get(metric_id)
        direction = _infer_direction(threshold_range)
        unit = threshold_range.unit if threshold_range else None
        method = threshold_range.elicitation_method if threshold_range else None
        notes = definition.definition or definition.justification
        hazard_link = authority_map.get(metric_id)

        metadata[metric_id] = MetricMetadata(
            id=metric_id,
            name=definition.name or metric_id,
            direction=direction,
            unit=unit or "value",
            method=method or "",
            hazard_link=hazard_link or "",
            notes=notes,
        )

    return metadata


def build_requirement_mapping_from_v2(
    mapping_entries: Sequence[MappingEntry],
    requirements_index: Mapping[str, RequirementRecord],
) -> Dict[str, RequirementMapping]:
    """Aggregate requirement references per metric for reporting."""

    grouped: Dict[str, set[str]] = {}
    category_hint: Dict[str, str | None] = {}

    for entry in mapping_entries:
        metric_id = entry.metric_id
        requirement_id = entry.requirement_id
        if not metric_id or not requirement_id:
            continue
        grouped.setdefault(metric_id, set()).add(requirement_id)
        requirement = requirements_index.get(requirement_id)
        if requirement and metric_id not in category_hint:
            category_hint[metric_id] = requirement.section or requirement.subsection

    mapping: Dict[str, RequirementMapping] = {}
    for metric_id, req_ids in grouped.items():
        mapping[metric_id] = RequirementMapping(
            category=category_hint.get(metric_id),
            requirements=", ".join(sorted(req_ids)),
            optimal_target=None,
        )

    return mapping


def build_threshold_tables_from_ranges(
    ranges: Mapping[str, ThresholdRange],
    metadata: Mapping[str, MetricMetadata],
) -> Dict[str, pd.DataFrame]:
    """Convert aggregated threshold ranges into a single table with min/max only."""

    records: Dict[str, Dict[str, object]] = {}
    for metric_id, range_obj in ranges.items():
        meta = metadata.get(metric_id)
        direction = meta.direction if meta else _infer_direction(range_obj)
        records[metric_id] = {
            "Metric_ID": metric_id,
            "Metric_Name": meta.name if meta else range_obj.metric_name or metric_id,
            "Unit": meta.unit if meta else range_obj.unit,
            "Direction": direction,
            "Threshold_Min": range_obj.min_value,
            "Threshold_Max": range_obj.max_value,
            "Assumptions": range_obj.source_category,
            "Method": range_obj.elicitation_method,
            "Rationale": range_obj.primary_source,
            "Enforce": range_obj.enforce_threshold,
        }

    return {"Ranges": pd.DataFrame.from_dict(records, orient="index")}


class AssuranceConfigV2:
    """Aggregates access to the revised configuration tables."""

    def __init__(self, config_root: Path) -> None:
        self.config_root = Path(config_root)
        self.base_dir = self.config_root / "config_v2"
        if not self.base_dir.exists():
            raise FileNotFoundError(f"config_v2 directory not found under {self.config_root}")

        self._catalog: Dict[str, MetricDefinition] | None = None
        self._threshold_ranges: Dict[str, ThresholdRange] | None = None
        self._test_beds: Dict[str, TestBedDefinition] | None = None
        self._mapping_entries: List[MappingEntry] | None = None
        self._requirements_index: Dict[str, RequirementRecord] | None = None
        self._metric_metadata: Dict[str, MetricMetadata] | None = None
        self._threshold_tables: Dict[str, pd.DataFrame] | None = None
        self._requirement_mapping: Dict[str, RequirementMapping] | None = None
        self._applicable_metric_ids: set[str] | None = None

    @property
    def catalog(self) -> Dict[str, MetricDefinition]:
        if self._catalog is None:
            catalog_path = self.base_dir / "C.Metric_catalog.csv"
            self._catalog = load_metric_catalog_v2(catalog_path)
        return self._catalog

    @property
    def threshold_ranges(self) -> Dict[str, ThresholdRange]:
        if self._threshold_ranges is None:
            thresholds_path = self.base_dir / "D.Thresholds.csv"
            self._threshold_ranges = load_threshold_ranges(thresholds_path)
        return self._threshold_ranges

    @property
    def test_beds(self) -> Dict[str, TestBedDefinition]:
        if self._test_beds is None:
            beds_path = self.base_dir / "E.Test_beds.csv"
            self._test_beds = load_test_beds(beds_path)
        return self._test_beds

    @property
    def mapping_entries(self) -> List[MappingEntry]:
        if self._mapping_entries is None:
            mapping_path = self.base_dir / "F.Mapping.csv"
            self._mapping_entries = load_mapping_entries(mapping_path)
        return self._mapping_entries

    @property
    def requirements_index(self) -> Dict[str, RequirementRecord]:
        if self._requirements_index is None:
            requirements_path = self.base_dir / "B.Requirements.csv"
            self._requirements_index = load_requirements_index(requirements_path)
        return self._requirements_index

    @property
    def metric_metadata(self) -> Dict[str, MetricMetadata]:
        if self._metric_metadata is None:
            catalog = {metric_id: definition for metric_id, definition in self.catalog.items() if metric_id in self.applicable_metric_ids}
            ranges = {metric_id: rng for metric_id, rng in self.threshold_ranges.items() if metric_id in catalog}
            metadata = build_metric_metadata_from_v2(
                catalog=catalog,
                thresholds=ranges,
                mappings=self.mapping_entries,
            )
            self._metric_metadata = metadata
        return self._metric_metadata

    @property
    def requirement_mapping(self) -> Dict[str, RequirementMapping]:
        if self._requirement_mapping is None:
            mapping = build_requirement_mapping_from_v2(
                mapping_entries=self.mapping_entries,
                requirements_index=self.requirements_index,
            )
            self._requirement_mapping = {
                metric_id: value
                for metric_id, value in mapping.items()
                if metric_id in self.applicable_metric_ids
            }
        return self._requirement_mapping

    @property
    def threshold_tables(self) -> Dict[str, pd.DataFrame]:
        if self._threshold_tables is None:
            filtered_ranges = {metric_id: rng for metric_id, rng in self.threshold_ranges.items() if metric_id in self.metric_metadata}
            self._threshold_tables = build_threshold_tables_from_ranges(
                ranges=filtered_ranges,
                metadata=self.metric_metadata,
            )
        return self._threshold_tables

    @property
    def applicable_metric_ids(self) -> set[str]:
        if self._applicable_metric_ids is None:
            applicable_path = self.base_dir / "C.1.1_Metrics_applicable.csv"
            applicable_definitions = load_metric_catalog_v2(applicable_path)
            self._applicable_metric_ids = set(applicable_definitions.keys())
        return self._applicable_metric_ids


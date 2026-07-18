"""Adversarial tests for the frozen scholarly-authority metadata gate."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.verify_authority_metadata import (
    MANIFEST_PATH,
    ROOT,
    SCHEMA_PATH,
    AuthorityMetadataError,
    FORMAL_EARLY_ACCESS_DOI,
    compare_bibtex_records,
    compare_catalogue_rows,
    compare_cff_preferred_citation,
    compare_standard_rows,
    load_manifest,
    parse_bibtex_records,
    parse_catalogue_rows,
    parse_standard_rows,
    validate_manifest,
    verify,
    verify_markdown_cutoffs,
)


def test_authority_snapshot_schema_scope_and_urls_are_valid() -> None:
    manifest = load_manifest()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert manifest["$schema"] == f"./{SCHEMA_PATH.name}"
    validate_manifest(manifest)


def test_json_schema_rejects_unreviewed_fields() -> None:
    manifest = copy.deepcopy(load_manifest())
    manifest["catalogue"]["records"][0]["invented_field"] = True

    with pytest.raises(AuthorityMetadataError, match="JSON Schema validation failed"):
        validate_manifest(manifest)


def test_every_catalogue_row_matches_frozen_crossref_metadata() -> None:
    manifest = load_manifest()
    rows = parse_catalogue_rows()

    compare_catalogue_rows(rows, manifest)
    assert len(rows) == 66
    assert len({row.doi for row in rows}) == 54


def test_all_tracked_bibtex_and_cff_records_match_authority() -> None:
    manifest = load_manifest()
    records = parse_bibtex_records()

    compare_bibtex_records(records, manifest)
    compare_cff_preferred_citation(manifest)
    assert len(records) == 9
    assert manifest["scope"]["unique_tracked_dois"] == 60
    assert manifest["scope"]["structured_citation_records"] == 66 + 9 + 1


def test_standardization_table_matches_official_record_snapshot() -> None:
    manifest = load_manifest()
    rows = parse_standard_rows()

    compare_standard_rows(rows, manifest)
    assert len(rows) == 12


def test_formal_early_access_record_is_frozen_and_disclosed() -> None:
    manifest = load_manifest()
    record = next(
        record
        for record in manifest["auxiliary_references"]["records"]
        if record["doi"] == FORMAL_EARLY_ACCESS_DOI
    )
    assert record["title"] == (
        "Simultaneous Sensing Data Acquisition and Sharing in Low-Altitude "
        "Wireless Networks: Fundamental Limits and Signaling Design"
    )
    assert [author["display"] for author in record["authors"]] == [
        "Fuwang Dong",
        "Fan Liu",
        "Yifeng Xiong",
        "Yuanhao Cui",
        "Wei Wang",
        "Shi Jin",
    ]
    assert record["venue"] == "IEEE Journal of Selected Topics in Signal Processing"
    assert record["year"] == 2026
    assert record["volume"] is None
    assert record["issue"] is None
    assert record["pages"] == "1-15"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        f"[{record['title']}]({record['doi_url']}) | "
        "IEEE JSTSP (Early Access) | 2026 |"
    ).casefold() in readme.casefold()
    network = (ROOT / "paper" / "network.md").read_text(encoding="utf-8")
    expected_network_record = (
        f"[{record['title']}]({record['doi_url']}) — {record['venue']}, "
        f"pp. {record['pages'].replace('-', '–')}, {record['year']}."
    )
    assert expected_network_record.casefold() in network.casefold()
    assert "had not deposited a volume or issue" in network


def test_markdown_discloses_snapshot_cutoff() -> None:
    manifest = load_manifest()
    verify_markdown_cutoffs(
        parse_catalogue_rows(), parse_standard_rows(), manifest
    )


@pytest.mark.parametrize(
    ("field", "fabricated"),
    [
        ("title", "Fabricated ISAC Result"),
        ("authors", ("Invented Author",)),
        ("venue", "Invented Transactions on ISAC"),
        ("year", 2099),
    ],
)
def test_single_doi_fabrication_is_rejected(field: str, fabricated: object) -> None:
    """A one-field adversarial edit cannot pass by remaining internally consistent."""
    manifest = load_manifest()
    rows = parse_catalogue_rows()
    index = next(
        index
        for index, row in enumerate(rows)
        if row.doi == "10.1109/twc.2024.3373797"
    )
    rows[index] = replace(rows[index], **{field: fabricated})

    with pytest.raises(AuthorityMetadataError, match=rf"{field} mismatch"):
        compare_catalogue_rows(rows, manifest)


def test_auxiliary_bibtex_fabrication_is_rejected() -> None:
    manifest = load_manifest()
    records = parse_bibtex_records()
    index = next(
        index
        for index, record in enumerate(records)
        if record.doi == "10.1109/jsen.2022.3208272"
    )
    records[index] = replace(records[index], title="Fabricated CSI Result")

    with pytest.raises(AuthorityMetadataError, match=r"BibTeX .* title mismatch"):
        compare_bibtex_records(records, manifest)


def test_offline_verifier_emits_hash_bound_evidence(tmp_path: Path) -> None:
    summary = verify()

    assert summary["result"] == "pass"
    assert summary["network_access_performed"] is False
    assert summary["snapshot"]["path"] == str(MANIFEST_PATH.relative_to(ROOT))
    assert len(summary["snapshot"]["sha256"]) == 64
    assert summary["catalogue"]["complete_required_fields"] == {
        "doi": 54,
        "title": 54,
        "authors": 54,
        "venue": 54,
        "year": 54,
        "volume": 54,
    }
    assert summary["catalogue"]["optional_field_coverage"] == {
        "issue": 49,
        "pages": 52,
        "article_number": 2,
    }
    assert summary["auxiliary_references"] == {
        "unique_dois": 6,
        "formal_early_access_dois": 1,
        "source_types": {"journal-article": 5, "monograph": 1},
    }
    assert summary["tracked_dois"] == 60
    assert summary["structured_citation_records"] == {
        "catalogue_rows": 66,
        "bibtex_records": 9,
        "cff_preferred_citations": 1,
        "total": 76,
        "exact_core_metadata_matches": {
            "doi": 76,
            "title": 76,
            "authors": 76,
            "venue_or_publisher": 76,
            "year": 76,
        },
        "exact_optional_field_matches": {
            "volume": 75,
            "issue": 66,
            "pages": 72,
            "article_number": 3,
        },
    }
    assert summary["standardization"]["rows"] == 12
    assert summary["authority_url_syntax_checks"] == 133

    # Keep the test's temporary output local while exercising JSON serialization.
    evidence = tmp_path / "authority-metadata-audit.json"
    evidence.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    assert json.loads(evidence.read_text(encoding="utf-8"))["result"] == "pass"

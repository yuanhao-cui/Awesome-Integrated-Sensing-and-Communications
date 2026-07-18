#!/usr/bin/env python3
"""Verify the frozen Crossref and official-standards metadata snapshots.

The verifier is deliberately offline.  Network retrieval happens only during an
explicit scholarly refresh; Gate 2 compares the reviewed, version-controlled
snapshot byte-for-byte (after limited Unicode and page-dash normalization) with
the catalogue and standards tables.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "audit" / "authority-metadata.json"
SCHEMA_PATH = ROOT / "audit" / "authority-metadata.schema.json"
HTTPS_PREFIX = "https:" + "//"
DOI_PREFIX = HTTPS_PREFIX + "doi.org/"
CROSSREF_WORKS_PREFIX = HTTPS_PREFIX + "api.crossref.org/works/"
DOI_LINK = re.compile(
    r"^\[([^]]+)]\(" + re.escape(DOI_PREFIX) + r"(10\.[^)]+)\)$", re.I
)
STANDARD_LINK = re.compile(
    r"^\[([^]]+)]\((" + re.escape(HTTPS_PREFIX) + r"[^)]+)\)$"
)
BIBLIOGRAPHIC_RECORD = re.compile(
    r"^(?P<venue>.+?), vol\. (?P<volume>[^,]+)"
    r"(?:, no\. (?P<issue>[^,]+))?, "
    r"(?:(?P<page_label>pp?\.) (?P<pages>[^,]+)|"
    r"Art\. no\. (?P<article_number>[^,]+)), "
    r"(?P<year>[0-9]{4})$"
)
DOI = re.compile(r"^10\.[0-9]{4,9}/[-._;()/:a-z0-9]+$")
ALLOWED_STANDARD_DOMAINS = {
    "IEEE": {"standards.ieee.org", "www.ieee802.org"},
    "3GPP": {"portal.3gpp.org"},
    "ITU-R": {"www.itu.int"},
    "ETSI": {"www.etsi.org"},
}
ALLOWED_MATURITIES = {
    "published-active-standard",
    "normative-technical-specification-under-change-control",
    "study-report-under-change-control",
    "draft-technical-specification",
    "in-force-recommendation",
    "pre-standardization-group",
    "published-group-report",
    "active-draft-project",
}


class AuthorityMetadataError(AssertionError):
    """Raised when tracked content diverges from the reviewed authority snapshot."""


@dataclass(frozen=True)
class CatalogueRow:
    path: str
    line: int
    doi: str
    title: str
    authors: tuple[str, ...]
    venue: str
    year: int
    volume: str
    issue: str | None
    pages: str | None
    article_number: str | None


@dataclass(frozen=True)
class StandardRow:
    path: str
    line: int
    organization: str
    identifier: str
    title: str
    official_url: str
    status: str
    maturity: str
    as_of: str
    evidence_scope: str


@dataclass(frozen=True)
class BibtexRecord:
    path: str
    line: int
    entry_type: str
    doi: str
    title: str
    authors: tuple[str, ...]
    venue: str | None
    publisher: str | None
    year: int
    volume: str | None
    issue: str | None
    pages: str | None


def clean_text(value: str) -> str:
    """Decode entities and normalize Unicode/whitespace without case-folding."""
    decoded = unicodedata.normalize("NFKC", html.unescape(value))
    return re.sub(r"\s+", " ", decoded).strip()


def normalize_pages(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = clean_text(value).translate(
        str.maketrans({"–": "-", "—": "-", "−": "-"})
    )
    return re.sub(r"-{2,}", "-", normalized)


def _https_url(value: str, *, label: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username:
        raise AuthorityMetadataError(f"{label}: not an absolute HTTPS URL: {value!r}")
    if any(char.isspace() for char in value):
        raise AuthorityMetadataError(f"{label}: URL contains whitespace: {value!r}")
    return parsed


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_catalogue_rows(root: Path = ROOT) -> list[CatalogueRow]:
    rows: list[CatalogueRow] = []
    for path in sorted((root / "paper").glob("*.md")):
        if path.name == "standardization.md":
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.lstrip().startswith("|") or "---" in line:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 3 or not (link := DOI_LINK.fullmatch(cells[0])):
                continue
            bibliographic = BIBLIOGRAPHIC_RECORD.fullmatch(cells[2])
            if bibliographic is None:
                raise AuthorityMetadataError(
                    f"{path.relative_to(root)}:{line_number}: cannot parse "
                    f"bibliographic record {cells[2]!r}"
                )
            fields = bibliographic.groupdict()
            authors = tuple(clean_text(author) for author in cells[1].split(";"))
            rows.append(
                CatalogueRow(
                    path=str(path.relative_to(root)),
                    line=line_number,
                    doi=link.group(2).casefold(),
                    title=clean_text(link.group(1)),
                    authors=authors,
                    venue=clean_text(fields["venue"]),
                    year=int(fields["year"]),
                    volume=fields["volume"],
                    issue=fields["issue"],
                    pages=normalize_pages(fields["pages"]),
                    article_number=fields["article_number"],
                )
            )
    return rows


def parse_standard_rows(root: Path = ROOT) -> list[StandardRow]:
    path = root / "paper" / "standardization.md"
    rows: list[StandardRow] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.lstrip().startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 7 or not (link := STANDARD_LINK.fullmatch(cells[1])):
            continue
        rows.append(
            StandardRow(
                path=str(path.relative_to(root)),
                line=line_number,
                organization=clean_text(cells[0]),
                identifier=clean_text(link.group(1)),
                title=clean_text(cells[2]),
                official_url=link.group(2),
                status=clean_text(cells[3]),
                maturity=clean_text(cells[4]),
                as_of=clean_text(cells[5]),
                evidence_scope=clean_text(cells[6]),
            )
        )
    return rows


def _bibtex_text(value: str) -> str:
    return clean_text(value.replace(r"\&", "&").replace("{", "").replace("}", ""))


def _bibtex_fields(entry: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    cursor = entry.find(",") + 1
    field_start = re.compile(r"\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*\{")
    while cursor > 0 and cursor < len(entry):
        match = field_start.match(entry, cursor)
        if match is None:
            cursor += 1
            continue
        depth = 1
        value_start = match.end()
        position = value_start
        while position < len(entry) and depth:
            if entry[position] == "{":
                depth += 1
            elif entry[position] == "}":
                depth -= 1
            position += 1
        if depth:
            raise AuthorityMetadataError("unterminated BibTeX field")
        fields[match.group(1).casefold()] = entry[value_start : position - 1]
        cursor = position
    return fields


def _bibtex_authors(value: str) -> tuple[str, ...]:
    authors = []
    for raw_author in re.split(r"\s+and\s+", clean_text(value)):
        if "," in raw_author:
            family, given = (part.strip() for part in raw_author.split(",", 1))
            authors.append(_bibtex_text(f"{given} {family}"))
        else:
            authors.append(_bibtex_text(raw_author))
    return tuple(authors)


def parse_bibtex_records(root: Path = ROOT) -> list[BibtexRecord]:
    records: list[BibtexRecord] = []
    paths = [root / "README.md", *sorted((root / "code").rglob("*.md"))]
    fence = re.compile(r"(?P<fence>```|~~~)bibtex\s*\n(.*?)(?P=fence)", re.I | re.S)
    entry_start = re.compile(r"@(article|book)\s*\{", re.I)
    for path in paths:
        content = path.read_text(encoding="utf-8")
        for block_match in fence.finditer(content):
            block = block_match.group(2)
            for start in entry_start.finditer(block):
                opening = start.end() - 1
                depth = 1
                position = opening + 1
                while position < len(block) and depth:
                    if block[position] == "{":
                        depth += 1
                    elif block[position] == "}":
                        depth -= 1
                    position += 1
                if depth:
                    raise AuthorityMetadataError(
                        f"{path.relative_to(root)}: unterminated BibTeX entry"
                    )
                entry = block[start.start() : position]
                fields = _bibtex_fields(entry)
                if "doi" not in fields:
                    continue
                entry_type = start.group(1).casefold()
                doi = _bibtex_text(fields["doi"]).casefold()
                line = content.count("\n", 0, block_match.start(2) + start.start()) + 1
                venue = fields.get("journal")
                publisher = fields.get("publisher")
                records.append(
                    BibtexRecord(
                        path=str(path.relative_to(root)),
                        line=line,
                        entry_type=entry_type,
                        doi=doi,
                        title=_bibtex_text(fields["title"]),
                        authors=_bibtex_authors(fields["author"]),
                        venue=_bibtex_text(venue) if venue is not None else None,
                        publisher=(
                            _bibtex_text(publisher)
                            if publisher is not None
                            else None
                        ),
                        year=int(_bibtex_text(fields["year"])),
                        volume=(
                            _bibtex_text(fields["volume"])
                            if "volume" in fields
                            else None
                        ),
                        issue=(
                            _bibtex_text(fields["number"])
                            if "number" in fields
                            else None
                        ),
                        pages=(
                            normalize_pages(_bibtex_text(fields["pages"]))
                            if "pages" in fields
                            else None
                        ),
                    )
                )
    return records


def _validate_crossref_record(
    record: dict[str, Any],
    *,
    allowed_source_types: set[str],
    venue_required: bool,
) -> None:
    doi = record.get("doi")
    if not isinstance(doi, str) or DOI.fullmatch(doi) is None:
        raise AuthorityMetadataError(f"invalid canonical DOI: {doi!r}")
    for field in ("title", "publisher", "source_type"):
        value = record.get(field)
        if not isinstance(value, str) or not clean_text(value):
            raise AuthorityMetadataError(f"{doi}: empty {field}")
    venue = record.get("venue")
    if venue_required and (not isinstance(venue, str) or not clean_text(venue)):
        raise AuthorityMetadataError(f"{doi}: empty venue")
    if venue is not None and (not isinstance(venue, str) or not clean_text(venue)):
        raise AuthorityMetadataError(f"{doi}: invalid venue")
    if record["source_type"] not in allowed_source_types:
        raise AuthorityMetadataError(
            f"{doi}: unexpected source_type {record['source_type']!r}"
        )
    authors = record.get("authors")
    if not isinstance(authors, list) or not authors:
        raise AuthorityMetadataError(f"{doi}: missing publication-order authors")
    for index, author in enumerate(authors, 1):
        expected_display = clean_text(
            f"{author.get('given', '')} {author.get('family', '')}"
        )
        if author.get("display") != expected_display:
            raise AuthorityMetadataError(
                f"{doi}: author {index} display does not match given/family"
            )
    if not isinstance(record.get("year"), int):
        raise AuthorityMetadataError(f"{doi}: year is not an integer")
    doi_url = DOI_PREFIX + doi
    source_url = CROSSREF_WORKS_PREFIX + urllib.parse.quote(
        doi, safe=""
    )
    if record.get("doi_url") != doi_url:
        raise AuthorityMetadataError(f"{doi}: non-canonical DOI URL")
    if record.get("source_url") != source_url:
        raise AuthorityMetadataError(f"{doi}: non-canonical Crossref source URL")
    _https_url(doi_url, label=f"{doi} doi_url")
    _https_url(source_url, label=f"{doi} source_url")
    try:
        datetime.fromisoformat(
            record["authority_record_updated_at"].replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityMetadataError(
            f"{doi}: invalid authority_record_updated_at"
        ) from exc


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Run JSON Schema 2020-12, then enforce scholarly semantic invariants."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise AuthorityMetadataError("authority schema is not JSON Schema 2020-12")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(manifest)
    except (SchemaError, ValidationError) as exc:
        location = "/".join(str(component) for component in exc.absolute_path)
        label = f" at {location}" if location else ""
        raise AuthorityMetadataError(
            f"authority metadata JSON Schema validation failed{label}: {exc.message}"
        ) from exc
    if manifest.get("$schema") != "./authority-metadata.schema.json":
        raise AuthorityMetadataError("manifest does not reference its local schema")
    if manifest.get("schema_version") != 1:
        raise AuthorityMetadataError("unsupported authority manifest schema_version")
    retrieved_at = manifest.get("retrieved_at")
    try:
        retrieval_date = date.fromisoformat(retrieved_at)
    except (TypeError, ValueError) as exc:
        raise AuthorityMetadataError(f"invalid retrieved_at: {retrieved_at!r}") from exc
    if retrieval_date != date(2026, 7, 18):
        raise AuthorityMetadataError(
            f"authority snapshot cutoff must be 2026-07-18, got {retrieved_at}"
        )

    scope = manifest.get("scope", {})
    expected_scope = {
        "catalogue_rows": 59,
        "unique_catalogue_dois": 49,
        "auxiliary_reference_dois": 5,
        "unique_tracked_dois": 54,
        "structured_citation_records": 69,
        "standardization_rows": 12,
    }
    if scope != expected_scope:
        raise AuthorityMetadataError(f"authority scope {scope!r} != {expected_scope!r}")
    if manifest.get("catalogue", {}).get("authority") != (
        "Crossref REST API publisher-deposited metadata"
    ):
        raise AuthorityMetadataError("catalogue authority must be Crossref")
    if manifest.get("auxiliary_references", {}).get("authority") != (
        "Crossref REST API publisher-deposited metadata"
    ):
        raise AuthorityMetadataError("auxiliary reference authority must be Crossref")
    if manifest.get("standardization", {}).get("authority") != (
        "Official standards-body records, kept separate from Crossref"
    ):
        raise AuthorityMetadataError("standards authority must remain separate from Crossref")
    authority_url = manifest["catalogue"].get("authority_url")
    if authority_url != "https://www.crossref.org/":
        raise AuthorityMetadataError(f"unexpected Crossref authority URL: {authority_url!r}")
    _https_url(authority_url, label="Crossref authority_url")

    records = manifest["catalogue"].get("records", [])
    if len(records) != 49:
        raise AuthorityMetadataError(f"expected 49 DOI records, found {len(records)}")
    dois = [record.get("doi") for record in records]
    if dois != sorted(dois) or len(dois) != len(set(dois)):
        raise AuthorityMetadataError("catalogue DOI records must be unique and sorted")
    for record in records:
        _validate_crossref_record(
            record,
            allowed_source_types={"journal-article"},
            venue_required=True,
        )

    auxiliary = manifest["auxiliary_references"].get("records", [])
    if len(auxiliary) != 5:
        raise AuthorityMetadataError(
            f"expected 5 auxiliary DOI records, found {len(auxiliary)}"
        )
    auxiliary_dois = [record.get("doi") for record in auxiliary]
    if auxiliary_dois != sorted(auxiliary_dois) or len(auxiliary_dois) != len(
        set(auxiliary_dois)
    ):
        raise AuthorityMetadataError("auxiliary DOI records must be unique and sorted")
    if set(dois) & set(auxiliary_dois):
        raise AuthorityMetadataError("catalogue and auxiliary DOI sets must be disjoint")
    if len(set(dois) | set(auxiliary_dois)) != 54:
        raise AuthorityMetadataError("tracked authority snapshot must contain 54 DOIs")
    for record in auxiliary:
        _validate_crossref_record(
            record,
            allowed_source_types={"journal-article", "monograph"},
            venue_required=record.get("source_type") == "journal-article",
        )
    if sum(record["source_type"] == "monograph" for record in auxiliary) != 1:
        raise AuthorityMetadataError("auxiliary snapshot must contain one monograph")

    standards = manifest["standardization"].get("records", [])
    if len(standards) != 12:
        raise AuthorityMetadataError(
            f"expected 12 official standards records, found {len(standards)}"
        )
    identifiers = [record.get("identifier") for record in standards]
    if len(identifiers) != len(set(identifiers)):
        raise AuthorityMetadataError("standard identifiers must be unique")
    for record in standards:
        identifier = record.get("identifier")
        for field in (
            "organization",
            "authority",
            "identifier",
            "title",
            "display_title",
            "status",
            "maturity",
            "official_url",
            "as_of",
            "evidence_scope",
        ):
            value = record.get(field)
            if not isinstance(value, str) or not clean_text(value):
                raise AuthorityMetadataError(f"{identifier}: empty {field}")
        if record["maturity"] not in ALLOWED_MATURITIES:
            raise AuthorityMetadataError(
                f"{identifier}: unsupported maturity {record['maturity']!r}"
            )
        if record["display_title"] != identifier:
            raise AuthorityMetadataError(
                f"{identifier}: display_title must equal the canonical identifier"
            )
        if record["as_of"] != retrieved_at:
            raise AuthorityMetadataError(
                f"{identifier}: as_of {record['as_of']} != {retrieved_at}"
            )
        parsed = _https_url(record["official_url"], label=f"{identifier} official_url")
        expected_domains = ALLOWED_STANDARD_DOMAINS.get(record["organization"])
        if expected_domains is None or parsed.hostname not in expected_domains:
            raise AuthorityMetadataError(
                f"{identifier}: {parsed.hostname!r} is not an official "
                f"{record['organization']} domain"
            )


def compare_catalogue_rows(
    rows: list[CatalogueRow], manifest: dict[str, Any]
) -> None:
    expected_records = {
        record["doi"]: record for record in manifest["catalogue"]["records"]
    }
    if len(rows) != manifest["scope"]["catalogue_rows"]:
        raise AuthorityMetadataError(
            f"expected {manifest['scope']['catalogue_rows']} catalogue rows, found {len(rows)}"
        )
    observed_dois = {row.doi for row in rows}
    if observed_dois != set(expected_records):
        missing = sorted(set(expected_records) - observed_dois)
        unexpected = sorted(observed_dois - set(expected_records))
        raise AuthorityMetadataError(
            f"catalogue DOI set mismatch; missing={missing}, unexpected={unexpected}"
        )
    for row in rows:
        expected = expected_records[row.doi]
        expected_fields: dict[str, object] = {
            "title": clean_text(expected["title"]),
            "authors": tuple(author["display"] for author in expected["authors"]),
            "venue": clean_text(expected["venue"]),
            "year": expected["year"],
            "volume": expected["volume"],
            "issue": expected["issue"],
            "pages": normalize_pages(expected["pages"]),
            "article_number": expected["article_number"],
        }
        for field, authoritative in expected_fields.items():
            observed = getattr(row, field)
            if observed != authoritative:
                raise AuthorityMetadataError(
                    f"{row.path}:{row.line}: {row.doi} {field} mismatch; "
                    f"observed={observed!r}, authority={authoritative!r}"
                )


def _crossref_records_by_doi(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = [
        *manifest["catalogue"]["records"],
        *manifest["auxiliary_references"]["records"],
    ]
    return {record["doi"]: record for record in records}


def compare_bibtex_records(
    records: list[BibtexRecord], manifest: dict[str, Any]
) -> None:
    if len(records) != 9:
        raise AuthorityMetadataError(f"expected 9 DOI-bearing BibTeX records, found {len(records)}")
    authority = _crossref_records_by_doi(manifest)
    expected_dois = {
        *(
            record["doi"]
            for record in manifest["auxiliary_references"]["records"]
        ),
        "10.1109/comst.2026.3655674",
        "10.1109/jsac.2022.3156632",
        "10.1109/tmc.2024.3462960",
    }
    observed_dois = {record.doi for record in records}
    if observed_dois != expected_dois:
        raise AuthorityMetadataError(
            "BibTeX DOI set mismatch; "
            f"missing={sorted(expected_dois - observed_dois)}, "
            f"unexpected={sorted(observed_dois - expected_dois)}"
        )
    for record in records:
        expected = authority[record.doi]
        expected_type = "book" if expected["source_type"] == "monograph" else "article"
        expected_fields: dict[str, object] = {
            "entry_type": expected_type,
            "title": clean_text(expected["title"]),
            "authors": tuple(author["display"] for author in expected["authors"]),
            "venue": clean_text(expected["venue"]) if expected["venue"] else None,
            "publisher": (
                expected["publisher"]
                if expected["source_type"] == "monograph"
                else None
            ),
            "year": expected["year"],
            "volume": expected["volume"],
            "issue": expected["issue"],
            "pages": normalize_pages(expected["pages"]),
        }
        for field, authoritative in expected_fields.items():
            observed = getattr(record, field)
            if observed != authoritative:
                raise AuthorityMetadataError(
                    f"{record.path}:{record.line}: BibTeX {record.doi} {field} "
                    f"mismatch; observed={observed!r}, authority={authoritative!r}"
                )


def compare_cff_preferred_citation(manifest: dict[str, Any], root: Path = ROOT) -> None:
    cff = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    preferred = cff["preferred-citation"]
    doi = preferred["doi"].casefold()
    expected = _crossref_records_by_doi(manifest).get(doi)
    if expected is None:
        raise AuthorityMetadataError(f"CITATION.cff DOI absent from authority snapshot: {doi}")
    pages = f"{preferred['start']}-{preferred['end']}"
    observed_fields = {
        "title": clean_text(preferred["title"]),
        "authors": tuple(
            clean_text(f"{author['given-names']} {author['family-names']}")
            for author in preferred["authors"]
        ),
        "venue": clean_text(preferred["journal"]),
        "year": int(preferred["year"]),
        "volume": str(preferred["volume"]),
        "pages": normalize_pages(pages),
    }
    expected_fields = {
        "title": clean_text(expected["title"]),
        "authors": tuple(author["display"] for author in expected["authors"]),
        "venue": clean_text(expected["venue"]),
        "year": expected["year"],
        "volume": expected["volume"],
        "pages": normalize_pages(expected["pages"]),
    }
    for field, authoritative in expected_fields.items():
        observed = observed_fields[field]
        if observed != authoritative:
            raise AuthorityMetadataError(
                f"CITATION.cff preferred-citation {doi} {field} mismatch; "
                f"observed={observed!r}, authority={authoritative!r}"
            )


def compare_standard_rows(rows: list[StandardRow], manifest: dict[str, Any]) -> None:
    standards = manifest["standardization"]["records"]
    if len(rows) != manifest["scope"]["standardization_rows"]:
        raise AuthorityMetadataError(
            f"expected {manifest['scope']['standardization_rows']} standards rows, "
            f"found {len(rows)}"
        )
    for row, expected in zip(rows, standards, strict=True):
        expected_fields = {
            "organization": expected["organization"],
            "identifier": expected["identifier"],
            "title": expected["title"],
            "official_url": expected["official_url"],
            "status": expected["status"],
            "maturity": expected["maturity"],
            "as_of": expected["as_of"],
            "evidence_scope": expected["evidence_scope"],
        }
        for field, authoritative in expected_fields.items():
            observed = getattr(row, field)
            if observed != authoritative:
                raise AuthorityMetadataError(
                    f"{row.path}:{row.line}: standard {expected['identifier']} "
                    f"{field} mismatch; observed={observed!r}, "
                    f"authority={authoritative!r}"
                )


def verify_markdown_cutoffs(
    catalogue_rows: list[CatalogueRow],
    standard_rows: list[StandardRow],
    manifest: dict[str, Any],
    root: Path = ROOT,
) -> None:
    cutoff = manifest["retrieved_at"]
    paths = {row.path for row in catalogue_rows} | {row.path for row in standard_rows}
    for relative in sorted(paths):
        content = (root / relative).read_text(encoding="utf-8")
        if cutoff not in content:
            raise AuthorityMetadataError(
                f"{relative}: does not disclose authority evidence cutoff {cutoff}"
            )


def verify(root: Path = ROOT, manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest)
    catalogue_rows = parse_catalogue_rows(root)
    standard_rows = parse_standard_rows(root)
    bibtex_records = parse_bibtex_records(root)
    compare_catalogue_rows(catalogue_rows, manifest)
    compare_standard_rows(standard_rows, manifest)
    compare_bibtex_records(bibtex_records, manifest)
    compare_cff_preferred_citation(manifest, root)
    verify_markdown_cutoffs(catalogue_rows, standard_rows, manifest, root)

    records = manifest["catalogue"]["records"]
    auxiliary = manifest["auxiliary_references"]["records"]
    authority = _crossref_records_by_doi(manifest)
    cff = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    cff_doi = cff["preferred-citation"]["doi"].casefold()
    structured_authority_records = [
        *(authority[row.doi] for row in catalogue_rows),
        *(authority[record.doi] for record in bibtex_records),
        authority[cff_doi],
    ]
    structured_optional_coverage = {
        field: sum(record[field] is not None for record in structured_authority_records)
        for field in ("volume", "issue", "pages", "article_number")
    }
    optional_coverage = {
        field: sum(record[field] is not None for record in records)
        for field in ("issue", "pages", "article_number")
    }
    snapshot_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    source_urls = {
        manifest["catalogue"]["authority_url"],
        *(record["doi_url"] for record in records),
        *(record["source_url"] for record in records),
        *(record["doi_url"] for record in auxiliary),
        *(record["source_url"] for record in auxiliary),
        *(record["official_url"] for record in manifest["standardization"]["records"]),
    }
    return {
        "schema_version": 1,
        "result": "pass",
        "network_access_performed": False,
        "snapshot": {
            "path": str(manifest_path.relative_to(root)),
            "sha256": snapshot_hash,
            "retrieved_at": manifest["retrieved_at"],
        },
        "catalogue": {
            "rows": len(catalogue_rows),
            "unique_dois": len({row.doi for row in catalogue_rows}),
            "complete_required_fields": {
                field: len(records)
                for field in ("doi", "title", "authors", "venue", "year", "volume")
            },
            "optional_field_coverage": optional_coverage,
        },
        "auxiliary_references": {
            "unique_dois": len(auxiliary),
            "source_types": {
                source_type: sum(
                    record["source_type"] == source_type for record in auxiliary
                )
                for source_type in ("journal-article", "monograph")
            },
        },
        "tracked_dois": len(_crossref_records_by_doi(manifest)),
        "structured_citation_records": {
            "catalogue_rows": len(catalogue_rows),
            "bibtex_records": len(bibtex_records),
            "cff_preferred_citations": 1,
            "total": len(catalogue_rows) + len(bibtex_records) + 1,
            "exact_core_metadata_matches": {
                field: len(structured_authority_records)
                for field in ("doi", "title", "authors", "venue_or_publisher", "year")
            },
            "exact_optional_field_matches": structured_optional_coverage,
        },
        "standardization": {"rows": len(standard_rows)},
        "authority_url_syntax_checks": len(source_urls),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, help="write deterministic audit evidence")
    args = parser.parse_args()
    summary = verify()
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

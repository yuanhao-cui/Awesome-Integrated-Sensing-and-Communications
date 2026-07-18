"""Deterministic scholarly-catalog and tracked-text integrity invariants."""

from __future__ import annotations

import re
import urllib.parse
from collections import defaultdict
from pathlib import Path

import yaml

from scripts.check_links import (
    extract_markdown_links,
    iter_markdown_files,
    tracked_repository_files,
)


ROOT = Path(__file__).resolve().parents[1]
DOI_URL = re.compile(r"^https://doi\.org/(10\.[^\s?#]+)$", re.I)
ARXIV_URL = re.compile(r"^https://arxiv\.org/abs/([^\s?#v]+)(?:v\d+)?$", re.I)
FORBIDDEN_DISCOVERY_URLS = (
    "scholar.google.",
    "webcache.googleusercontent.com",
    "google.com/search",
    "bing.com/search",
)
PLACEHOLDER_TITLE = re.compile(r"(?:\.\.\.|\b(?:todo|tbd|placeholder)\b|🚧)", re.I)


def _normalize_title(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[*_`{}]", "", value)
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _paper_rows():
    for path in sorted((ROOT / "paper").glob("*.md")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.lstrip().startswith("|") or "---" in line:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            match = re.fullmatch(r"\[([^]]+)]\(([^)]+)\)", cells[0])
            if match:
                yield (
                    path,
                    line_number,
                    match.group(1),
                    match.group(2),
                    cells[1],
                    cells[2],
                )


def _canonical_identifier(target: str) -> str | None:
    if match := DOI_URL.match(target):
        return "doi:" + match.group(1).casefold().rstrip(".")
    if match := ARXIV_URL.match(target):
        return "arxiv:" + match.group(1).casefold()
    return None


def test_paper_entries_use_canonical_scholarly_sources():
    failures = []
    count = 0
    for path, line, title, target, _authors, _metadata in _paper_rows():
        count += 1
        if _canonical_identifier(target) is None:
            failures.append(f"{path.relative_to(ROOT)}:{line}: {target}")
        if PLACEHOLDER_TITLE.search(title):
            failures.append(f"{path.relative_to(ROOT)}:{line}: placeholder title {title!r}")
    assert count > 0, "No scholarly paper rows were discovered"
    assert not failures, "Non-canonical scholarly rows:\n" + "\n".join(failures)


def test_readme_preserves_the_original_featured_paper_structure():
    """Keep all 44 original rows and freeze the nine reviewed additions."""
    expected_rows = {
        "🔥 Landmark Surveys": 12,
        "📡 RF ISAC — Antenna & Waveform": 9,
        "🔦 Optical ISAC": 6,
        "🌐 Network Architecture": 10,
        "🧠 AI/ML for ISAC": 10,
        "🔒 Security": 6,
    }
    original_identifiers = {
        "doi:10.1109/comst.2026.3655674",
        "doi:10.1109/comst.2026.3670854",
        "doi:10.1109/jsac.2022.3156632",
        "arxiv:2510.14358",
        "doi:10.1109/msp.2023.3272881",
        "doi:10.1109/jstsp.2021.3113120",
        "doi:10.1109/tcomm.2020.2973976",
        "doi:10.16798/j.issn.1003-0530.2024.08.001",
        "doi:10.1109/jsac.2020.3007211",
        "doi:10.1109/tvt.2023.3248657",
        "doi:10.1109/globecom52923.2024.10901440",
        "doi:10.1038/s41467-025-57137-6",
        "doi:10.1109/mwc.001.2400201",
        "doi:10.1109/twc.2020.2998583",
        "doi:10.1109/mwc.018.2300607",
        "doi:10.1109/twc.2024.3503605",
        "doi:10.1109/iotm.001.2300196",
        "doi:10.1364/ofc.2024.tu3k.4",
        "doi:10.1109/jlt.2024.3392282",
        "doi:10.1109/tmtt.2024.3355936",
        "doi:10.1364/oe.474055",
        "doi:10.1038/s41377-022-01067-1",
        "doi:10.1109/mwc.008.2400151",
        "doi:10.1109/mcom.001.2300674",
        "doi:10.1109/iotm.001.2400042",
        "doi:10.1109/twc.2024.3417713",
        "doi:10.1109/twc.2023.3325849",
        "doi:10.1109/mcom.008.2200510",
        "doi:10.1109/twc.2022.3211533",
        "doi:10.1109/mcom.007.2200459",
        "doi:10.1109/mcom.001.2300660",
        "doi:10.1109/jstsp.2022.3226836",
        "doi:10.1109/mcom.001.2300724",
        "doi:10.1109/mnet.2023.3326064",
        "doi:10.23919/jcin.2022.9906941",
        "doi:10.1109/tcomm.2024.3375818",
        "doi:10.1109/comst.2023.3336917",
        "doi:10.1145/3638550.3641130",
        "doi:10.1109/twc.2020.3023164",
        "doi:10.1109/twc.2022.3156893",
        "doi:10.1109/tvt.2024.3422036",
        "doi:10.1109/iotm.001.2300180",
        "doi:10.1109/tifs.2024.3414185",
        "doi:10.1109/lwc.2024.3434470",
    }
    added_identifiers_by_section = {
        "🔥 Landmark Surveys": {
            "doi:10.1109/comst.2022.3149272",
            "doi:10.1109/comst.2021.3122519",
            "doi:10.1109/jiot.2023.3235618",
            "doi:10.1109/jiot.2026.3693228",
            "doi:10.1109/jsac.2025.3614025",
        },
        "🌐 Network Architecture": {
            "doi:10.1109/tmc.2025.3581616",
            "doi:10.1109/jstsp.2026.3696543",
        },
        "🧠 AI/ML for ISAC": {
            "doi:10.1109/jstsp.2024.3405859",
            "doi:10.1109/tmc.2026.3674960",
        },
    }
    observed_rows = defaultdict(int)
    observed_identifiers = set()
    section_identifiers = defaultdict(list)
    current_section = None
    in_featured = False
    failures = []

    for line_number, line in enumerate(
        (ROOT / "README.md").read_text(encoding="utf-8").splitlines(), 1
    ):
        if line == "## ⭐ Featured Papers":
            in_featured = True
            continue
        if in_featured and line.startswith("## "):
            break
        if not in_featured:
            continue
        if line.startswith("### "):
            current_section = line.removeprefix("### ")
            continue
        if not line.startswith("| ["):
            continue
        if current_section not in expected_rows:
            failures.append(
                f"README.md:{line_number}: featured row outside an expected section"
            )
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        match = re.fullmatch(r"\[([^]]+)]\(([^)]+)\)", cells[0])
        if match is None:
            failures.append(f"README.md:{line_number}: malformed publication link")
            continue
        title, target = match.groups()
        identifier = _canonical_identifier(target)
        if identifier is None:
            failures.append(f"README.md:{line_number}: non-canonical source {target}")
        else:
            observed_identifiers.add(identifier)
            section_identifiers[current_section].append(identifier)
        if PLACEHOLDER_TITLE.search(title):
            failures.append(f"README.md:{line_number}: placeholder title {title!r}")
        observed_rows[current_section] += 1

    assert dict(observed_rows) == expected_rows
    assert sum(observed_rows.values()) == 53
    assert len(observed_identifiers) == 53
    assert original_identifiers <= observed_identifiers
    assert len(original_identifiers) == 44
    added_identifiers = set().union(*added_identifiers_by_section.values())
    assert len(added_identifiers) == 9
    assert observed_identifiers == original_identifiers | added_identifiers
    for section, identifiers in added_identifiers_by_section.items():
        assert identifiers <= set(section_identifiers[section])
    assert section_identifiers["🔥 Landmark Surveys"][-1] == "arxiv:2510.14358"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Featured-53-orange" in readme
    assert "A targeted review of the maintainer's recent publications" in " ".join(
        readme.split()
    )
    assert not failures, "Invalid featured-paper rows:\n" + "\n".join(failures)


def test_identifier_title_and_metadata_are_one_to_one():
    by_identifier = defaultdict(set)
    by_title = defaultdict(set)
    authors_by_title = defaultdict(set)
    metadata_by_title = defaultdict(set)
    locations = defaultdict(list)
    for path, line, title, target, authors, metadata in _paper_rows():
        identifier = _canonical_identifier(target)
        if identifier is None:
            continue
        normalized_title = _normalize_title(title)
        by_identifier[identifier].add(normalized_title)
        by_title[normalized_title].add(identifier)
        authors_by_title[normalized_title].add(_normalize_title(authors))
        metadata_by_title[normalized_title].add(_normalize_title(metadata))
        locations[(identifier, normalized_title)].append(
            f"{path.relative_to(ROOT)}:{line}"
        )

    conflicts = []
    for identifier, titles in by_identifier.items():
        if len(titles) > 1:
            conflicts.append(f"{identifier} -> {sorted(titles)}")
    for title, identifiers in by_title.items():
        if len(identifiers) > 1:
            conflicts.append(f"{title!r} -> {sorted(identifiers)}")
        if len(authors_by_title[title]) > 1:
            conflicts.append(
                f"{title!r} has inconsistent author order: "
                f"{sorted(authors_by_title[title])}"
            )
        if len(metadata_by_title[title]) > 1:
            conflicts.append(
                f"{title!r} has inconsistent venue/year metadata: "
                f"{sorted(metadata_by_title[title])}"
            )
    assert not conflicts, "Catalog identity conflicts:\n" + "\n".join(conflicts)


def test_no_search_cache_or_placeholder_documentation_urls():
    failures = []
    for path in iter_markdown_files(ROOT):
        for reference in extract_markdown_links(path):
            lowered = reference.target.casefold()
            if any(marker in lowered for marker in FORBIDDEN_DISCOVERY_URLS):
                failures.append(
                    f"{path.relative_to(ROOT)}:{reference.line}: {reference.target}"
                )
            if "..." in reference.target or "{url}" in lowered:
                failures.append(
                    f"{path.relative_to(ROOT)}:{reference.line}: placeholder {reference.target}"
                )
    assert not failures, "Forbidden documentation URLs:\n" + "\n".join(failures)


def test_markdown_table_column_counts_are_consistent():
    failures = []
    for path in iter_markdown_files(ROOT):
        expected = None
        in_fence = False
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
                expected = None
                continue
            if in_fence or not line.lstrip().startswith("|"):
                expected = None
                continue
            columns = len(line.strip().strip("|").split("|"))
            if expected is None:
                expected = columns
            elif columns != expected:
                failures.append(
                    f"{path.relative_to(ROOT)}:{line_number}: {columns} columns, expected {expected}"
                )
    assert not failures, "Malformed Markdown tables:\n" + "\n".join(failures)


def test_cff_preferred_citation_matches_readme():
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    preferred = cff["preferred-citation"]
    assert [
        (author["given-names"], author["family-names"])
        for author in cff["authors"]
    ] == [("Yuanhao", "Cui")]
    expected_publication_authors = [
        ("Di", "Zhang"),
        ("Yuanhao", "Cui"),
        ("Xiaowen", "Cao"),
        ("Nanchi", "Su"),
        ("Yi", "Gong"),
        ("Fan", "Liu"),
        ("Weijie", "Yuan"),
        ("Xiaojun", "Jing"),
        ("J. Andrew", "Zhang"),
        ("Jie", "Xu"),
        ("Christos", "Masouros"),
        ("Dusit", "Niyato"),
        ("Marco", "Di Renzo"),
    ]
    assert [
        (author["given-names"], author["family-names"])
        for author in preferred["authors"]
    ] == expected_publication_authors
    assert cff["license"] == "CC-BY-SA-4.0"
    assert cff["repository-code"] == (
        "https://github.com/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications"
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        preferred["doi"],
        preferred["journal"],
        preferred["volume"],
        preferred["start"],
        preferred["end"],
        preferred["year"],
    ) == (
        "10.1109/COMST.2026.3655674",
        "IEEE Communications Surveys & Tutorials",
        28,
        5014,
        5048,
        2026,
    )
    assert preferred["title"] in readme
    assert f"https://doi.org/{preferred['doi']}" in readme


def test_tracked_text_http_urls_are_well_formed():
    """Inventory URLs in docs, metadata, workflows, and executable source."""
    suffixes = {".md", ".yml", ".yaml", ".json", ".cff", ".py", ".toml"}
    url_pattern = re.compile(r"https?://[^\s<>\"']+")
    failures = []
    count = 0
    for path in tracked_repository_files(ROOT, suffixes):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in url_pattern.finditer(line):
                count += 1
                target = match.group(0).rstrip(".,;:!?)]}")
                parsed = urllib.parse.urlsplit(target)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    failures.append(f"{path.relative_to(ROOT)}:{line_number}: {target}")
    assert count >= 100, f"Suspiciously small tracked-link inventory: {count}"
    assert not failures, "Malformed tracked-text URLs:\n" + "\n".join(failures)


def test_live_link_exceptions_are_exact_and_match_workflow():
    registry = yaml.safe_load(
        (ROOT / "link-exceptions.yaml").read_text(encoding="utf-8")
    )
    expected_urls = {
        "https://export.arxiv.org/api/query",
        "http://www.w3.org/2005/Atom",
        "https://www.mathworks.com/products/radar.html",
        "https://doi.org/10.1002/0471663085",
        "https://doi.org/10.1145/3638550.3641130",
    }
    entries = registry["exceptions"]
    assert registry["schema_version"] == 1
    assert str(registry["verified_on"]) == "2026-07-18"
    assert {entry["url"] for entry in entries} == expected_urls
    patterns = [entry["lychee_pattern"] for entry in entries]
    assert len(patterns) == len(set(patterns)) == 5
    assert all(pattern.startswith("^") and pattern.endswith("$") for pattern in patterns)

    workflow = (ROOT / ".github/workflows/link-check.yml").read_text(
        encoding="utf-8"
    )
    configured = set(re.findall(r"--exclude '([^']+)'", workflow))
    assert configured == set(patterns)
    configured_paths = set(re.findall(r"--exclude-path '([^']+)'", workflow))
    assert configured_paths == {
        r"(^|/)\.pytest_cache/",
        r"(^|/)link-exceptions\.yaml$",
        r"(^|/)\.github/workflows/link-check\.yml$",
    }
    assert "--accept" not in workflow
    assert "--cache" not in workflow
    assert "actions/cache" not in workflow
    assert "--host-concurrency 2" in workflow
    assert "--host-request-interval 250ms" in workflow
    assert "--max-concurrency 12" in workflow
    assert "'./audit/*.json'" in workflow

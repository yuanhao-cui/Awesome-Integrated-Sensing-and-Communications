#!/usr/bin/env python3
"""List newly submitted ISAC papers from the public arXiv Atom API.

The network fetch and the deterministic filtering/deduplication steps are kept
separate so a failed query can never be presented as a complete result set.
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

ARXIV_API = "https://export.arxiv.org/api/query"
USER_AGENT = "awesome-isac-audit/2.0 (+https://github.com/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications)"
ISAC_QUERIES = (
    'all:"integrated sensing" AND all:communication',
    'all:"joint radar" AND all:communication',
    'all:"dual functional radar" AND all:communication',
)
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def build_query_url(query: str, max_results: int = 50) -> str:
    """Build a safely encoded arXiv API URL."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if max_results < 1 or max_results > 2_000:
        raise ValueError("max_results must be between 1 and 2000")
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        }
    )
    return f"{ARXIV_API}?{params}"


def fetch_arxiv(
    query: str,
    max_results: int = 50,
    timeout: float = 30.0,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> str:
    """Fetch one arXiv query and return its UTF-8 Atom payload."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    request = urllib.request.Request(
        build_query_url(query, max_results),
        headers={"User-Agent": USER_AGENT, "Accept": "application/atom+xml"},
    )
    with opener(request, timeout=timeout) as response:
        payload = response.read()
    return payload.decode("utf-8")


def _required_text(entry: ET.Element, tag: str, entry_number: int) -> str:
    element = entry.find(tag, ATOM_NS)
    text = element.text.strip() if element is not None and element.text else ""
    if not text:
        raise ValueError(f"arXiv entry {entry_number} is missing {tag!r}")
    return text


def parse_entries(xml_data: str) -> list[dict[str, object]]:
    """Parse Atom entries, rejecting malformed records explicitly."""
    root = ET.fromstring(xml_data)
    entries: list[dict[str, object]] = []
    for index, entry in enumerate(root.findall("atom:entry", ATOM_NS), start=1):
        title = " ".join(_required_text(entry, "atom:title", index).split())
        link = _required_text(entry, "atom:id", index)
        published = _required_text(entry, "atom:published", index)
        parse_timestamp(published)  # validate at the ingestion boundary
        authors = [
            " ".join(author.text.split())
            for author in entry.findall("atom:author/atom:name", ATOM_NS)
            if author.text and author.text.strip()
        ]
        if not authors:
            raise ValueError(f"arXiv entry {index} has no authors")
        entries.append(
            {
                "title": title,
                "link": link,
                "published": published,
                "authors": authors,
            }
        )
    return entries


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to aware UTC."""
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid arXiv timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"arXiv timestamp lacks a timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def canonical_arxiv_id(link: str) -> str:
    """Return a version-independent identifier for an arXiv entry URL."""
    parsed = urllib.parse.urlsplit(link)
    match = re.search(r"/(?:abs|pdf)/([^/?#]+)", parsed.path)
    identifier = match.group(1) if match else parsed.path.rsplit("/", 1)[-1]
    identifier = re.sub(r"\.pdf$", "", identifier, flags=re.IGNORECASE)
    return re.sub(r"v\d+$", "", identifier, flags=re.IGNORECASE).lower()


def filter_recent_entries(
    entries: Iterable[dict[str, object]],
    days: int,
    now: datetime | None = None,
) -> list[dict[str, object]]:
    """Filter by the inclusive UTC submission cutoff."""
    if days < 0:
        raise ValueError("days must be non-negative")
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    reference = reference.astimezone(timezone.utc)
    cutoff = reference - timedelta(days=days)
    recent = []
    for entry in entries:
        published = parse_timestamp(str(entry["published"]))
        if cutoff <= published <= reference:
            recent.append(entry)
    return recent


def deduplicate_and_sort(
    entries: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Deduplicate versioned IDs and sort newest-first deterministically."""
    ordered = sorted(
        entries,
        key=lambda entry: (
            parse_timestamp(str(entry["published"])),
            canonical_arxiv_id(str(entry["link"])),
            str(entry["title"]).casefold(),
        ),
        reverse=True,
    )
    unique: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in ordered:
        identifier = canonical_arxiv_id(str(entry["link"]))
        if identifier not in seen:
            seen.add(identifier)
            unique.append(entry)
    return unique


def _format_authors(authors: object, limit: int = 3) -> str:
    names = list(authors) if isinstance(authors, list) else []
    return ", ".join(str(name) for name in names[:limit]) + (
        "..." if len(names) > limit else ""
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--max", dest="max_results", type=int, default=50)
    args = parser.parse_args(argv)
    if args.days < 0:
        parser.error("--days must be non-negative")
    if not 1 <= args.max_results <= 2_000:
        parser.error("--max must be between 1 and 2000")

    all_entries: list[dict[str, object]] = []
    for query in ISAC_QUERIES:
        try:
            all_entries.extend(parse_entries(fetch_arxiv(query, args.max_results)))
        except (OSError, TimeoutError, UnicodeError, ValueError, ET.ParseError) as exc:
            print(f"ERROR: arXiv query failed ({query!r}): {exc}", file=sys.stderr)
            print("No completeness claim has been made; retry later.", file=sys.stderr)
            return 2

    recent = filter_recent_entries(all_entries, args.days)
    unique = deduplicate_and_sort(recent)
    print(f"Found {len(unique)} ISAC paper(s) submitted in the last {args.days} day(s).")
    for entry in unique:
        published = parse_timestamp(str(entry["published"])).date().isoformat()
        print(f"\n[{published}] {entry['title']}")
        print(f"  {entry['link']}")
        print(f"  Authors: {_format_authors(entry['authors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

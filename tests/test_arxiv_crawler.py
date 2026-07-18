"""Deterministic tests for the arXiv ingestion pipeline."""

from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts.arxiv_crawler import (
    ARXIV_API,
    build_query_url,
    canonical_arxiv_id,
    deduplicate_and_sort,
    fetch_arxiv,
    filter_recent_entries,
    parse_entries,
)


def _feed(*entries: tuple[str, str, str, str]) -> str:
    body = []
    for title, link, published, author in entries:
        body.append(
            f"""
            <entry>
              <title>{title}</title>
              <id>{link}</id>
              <published>{published}</published>
              <author><name>{author}</name></author>
            </entry>
            """
        )
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        + "".join(body)
        + "</feed>"
    )


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self.payload


def test_fetch_uses_https_and_url_encoding():
    observed = {}

    def opener(request, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return _Response(_feed().encode())

    assert fetch_arxiv('all:"joint radar" AND all:communication', opener=opener) == _feed()
    parsed = urlsplit(observed["url"])
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == ARXIV_API
    assert parse_qs(parsed.query)["search_query"] == [
        'all:"joint radar" AND all:communication'
    ]
    assert observed["timeout"] == 30.0


def test_days_filter_is_utc_aware_and_inclusive():
    now = datetime(2026, 7, 17, 12, tzinfo=timezone.utc)
    entries = parse_entries(
        _feed(
            ("at cutoff", "https://arxiv.org/abs/2607.00001", "2026-07-10T12:00:00Z", "A"),
            ("too old", "https://arxiv.org/abs/2607.00002", "2026-07-10T11:59:59Z", "B"),
            ("recent", "https://arxiv.org/abs/2607.00003", "2026-07-17T11:00:00+00:00", "C"),
            ("future", "https://arxiv.org/abs/2607.00004", "2026-07-17T12:00:01Z", "D"),
        )
    )
    assert [entry["title"] for entry in filter_recent_entries(entries, 7, now)] == [
        "at cutoff",
        "recent",
    ]


def test_versioned_duplicates_are_removed_and_results_are_stably_sorted():
    entries = parse_entries(
        _feed(
            ("older version", "http://arxiv.org/abs/2607.00001v1", "2026-07-15T00:00:00Z", "A"),
            ("newer version", "https://arxiv.org/abs/2607.00001v2", "2026-07-16T00:00:00Z", "A"),
            ("other", "https://export.arxiv.org/abs/2607.00002", "2026-07-17T00:00:00Z", "B"),
        )
    )
    unique = deduplicate_and_sort(entries)
    assert [entry["title"] for entry in unique] == ["other", "newer version"]
    assert canonical_arxiv_id("https://arxiv.org/pdf/2607.00001v3.pdf") == "2607.00001"


@pytest.mark.parametrize("max_results", [0, 2001])
def test_query_limit_validation(max_results):
    with pytest.raises(ValueError, match="between 1 and 2000"):
        build_query_url("all:isac", max_results)


def test_malformed_entry_fails_explainably():
    malformed = _feed(("title", "https://arxiv.org/abs/2607.1", "not-a-date", "A"))
    with pytest.raises(ValueError, match="invalid arXiv timestamp"):
        parse_entries(malformed)

"""Offline unit tests for the explainable link auditor (no live network)."""

import urllib.error

from scripts.check_links import (
    audit_offline,
    check_url,
    extract_markdown_links,
)


class _Response:
    def __init__(self, status=200, url="https://example.org/final", body=b""):
        self.status = status
        self.url = url
        self.body = body

    def read(self, size=-1):
        return self.body[:size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _http_error(request, code):
    return urllib.error.HTTPError(request.full_url, code, "test status", {}, None)


def test_extraction_handles_balanced_parentheses_and_bare_punctuation(tmp_path):
    markdown = tmp_path / "sample.md"
    markdown.write_text(
        "[Paper](https://example.org/a_(b))\n"
        "Bare: https://example.org/report.pdf.\n"
        "[Local](guide.md#method)\n",
        encoding="utf-8",
    )
    references = extract_markdown_links(markdown)
    assert [(reference.label, reference.target) for reference in references] == [
        ("Paper", "https://example.org/a_(b)"),
        ("https://example.org/report.pdf", "https://example.org/report.pdf"),
        ("Local", "guide.md#method"),
    ]


def test_offline_audit_checks_local_paths_and_anchors(tmp_path):
    readme = tmp_path / "README.md"
    guide = tmp_path / "guide.md"
    guide.write_text("# Method\n", encoding="utf-8")
    readme.write_text(
        "[ok](guide.md#method) [missing](guide.md#unknown) "
        "[file](absent.md) [bad](https://example.com/...)\n",
        encoding="utf-8",
    )
    failures = audit_offline(tmp_path, extract_markdown_links(readme))
    assert [detail for _, detail in failures] == [
        "missing Markdown anchor #unknown",
        "missing local target 'absent.md'",
        "placeholder URL",
    ]


def test_head_not_supported_falls_back_to_bounded_get():
    methods = []

    def opener(request, timeout):
        methods.append(request.get_method())
        if request.get_method() == "HEAD":
            raise _http_error(request, 405)
        assert request.headers["Range"] == "bytes=0-8191"
        return _Response()

    result = check_url("https://example.org/paper", retries=0, opener=opener)
    assert result.category == "ok"
    assert methods == ["HEAD", "GET"]


def test_not_found_is_conclusive_but_forbidden_is_not():
    def not_found(request, timeout):
        raise _http_error(request, 404)

    assert check_url("https://example.org/missing", retries=0, opener=not_found).category == "broken"

    def forbidden(request, timeout):
        raise _http_error(request, 403)

    blocked = check_url("https://example.org/blocked", retries=0, opener=forbidden)
    assert blocked.category == "blocked"
    assert not blocked.healthy


def test_transient_status_is_retried_without_becoming_false_broken():
    calls = 0
    sleeps = []

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(request, 503)
        return _Response()

    result = check_url(
        "https://example.org/temporary",
        retries=1,
        opener=opener,
        sleeper=sleeps.append,
    )
    assert result.category == "ok"
    assert result.attempts == 2
    assert sleeps == [0.25]


def test_nested_badge_extracts_image_and_outer_destination(tmp_path):
    markdown = tmp_path / "badge.md"
    markdown.write_text(
        "[![Build](https://img.example.org/build.svg)]"
        "(https://github.com/example/project/actions)\n",
        encoding="utf-8",
    )
    targets = [ref.target for ref in extract_markdown_links(markdown)]
    assert set(targets) == {
        "https://github.com/example/project/actions",
        "https://img.example.org/build.svg",
    }
    assert len(targets) == 2


def test_html_soft_404_is_broken_even_with_http_200():
    def opener(request, timeout):
        return _Response(
            status=200,
            url="https://vendor.example.org/errors/404.html",
            body=b"<html><title>Page not found</title></html>",
        )

    result = check_url("https://vendor.example.org/old", retries=0, opener=opener)
    assert result.category == "broken"
    assert "soft-404" in result.detail

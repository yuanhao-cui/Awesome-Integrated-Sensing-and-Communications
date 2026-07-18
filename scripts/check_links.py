#!/usr/bin/env python3
"""Audit repository documentation links with explainable offline/online results.

Offline validation is deterministic: malformed URLs, missing local targets, and
missing Markdown anchors are conclusive failures.  Online validation retries
transient failures and distinguishes broken (for example 404/410), blocked
(401/403), rate-limited (429), and transport failures.  An uncertain network
result is never reported as either healthy or broken.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import re
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

USER_AGENT = "awesome-isac-audit/2.0 (+https://github.com/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications)"
EXCLUDED_DIRECTORIES = {".git", ".pytest_cache", "__pycache__", ".venv", "venv"}
PLACEHOLDER_MARKERS = ("...", "{url}", "<url>", "example.com", "placeholder")
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
BLOCKED_STATUS = {401, 403, 407, 451}
BROKEN_STATUS = {400, 404, 410, 414, 422}


@dataclasses.dataclass(frozen=True)
class LinkReference:
    source: Path
    line: int
    label: str
    target: str


@dataclasses.dataclass(frozen=True)
class LinkResult:
    url: str
    category: str
    status: int | None
    detail: str
    attempts: int
    final_url: str | None = None

    @property
    def healthy(self) -> bool:
        return self.category == "ok"


def tracked_repository_files(
    repo_root: Path,
    suffixes: set[str] | None = None,
) -> list[Path]:
    """Return tracked plus non-ignored untracked files deterministically.

    Including non-ignored untracked files makes this safe during local review,
    while avoiding generated, ignored virtual environments and caches.
    """
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode == 0:
        relative_paths = [
            Path(raw.decode("utf-8"))
            for raw in result.stdout.split(b"\0")
            if raw
        ]
        candidates = [repo_root / path for path in relative_paths]
    else:
        candidates = [
            path
            for path in repo_root.rglob("*")
            if not any(part in EXCLUDED_DIRECTORIES for part in path.parts)
        ]
    return sorted(
        path
        for path in candidates
        if path.is_file()
        and (suffixes is None or path.suffix.casefold() in suffixes)
    )


def iter_markdown_files(repo_root: Path) -> list[Path]:
    """Return repository Markdown source files in deterministic order."""
    return tracked_repository_files(repo_root, {".md"})


def _trim_bare_url(url: str) -> str:
    url = url.rstrip(".,;:!?")
    pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    changed = True
    while changed and url:
        changed = False
        for opening, closing in pairs:
            if url.endswith(closing) and url.count(closing) > url.count(opening):
                url = url[:-1]
                changed = True
    return url


def _inline_links(content: str) -> list[tuple[int, int, str, str]]:
    """Extract inline Markdown destinations, including balanced parentheses."""
    matches: list[tuple[int, int, str, str]] = []
    opener = re.compile(r"!?\[([^\]\n]*)\]\(")
    for match in opener.finditer(content):
        cursor = match.end()
        if cursor >= len(content):
            continue
        if content[cursor] == "<":
            closing = content.find(">", cursor + 1)
            if closing == -1:
                continue
            target = content[cursor + 1 : closing]
            end = closing + 1
        else:
            start = cursor
            depth = 0
            escaped = False
            while cursor < len(content):
                character = content[cursor]
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == "(":
                    depth += 1
                elif character == ")":
                    if depth == 0:
                        break
                    depth -= 1
                elif character.isspace() and depth == 0:
                    break
                cursor += 1
            target = content[start:cursor].replace("\\)", ")")
            end = cursor
        if target:
            matches.append((match.start(), end, match.group(1), target))
    return matches


def _outer_image_links(content: str) -> list[tuple[int, int, str, str]]:
    """Extract the destination wrapping a Markdown image/badge.

    The ordinary inline scanner sees the inner ``![alt](image)`` first.  This
    second, deliberately narrow pass captures ``[![alt](image)](destination)``
    without pretending to be a full Markdown parser.
    """
    pattern = re.compile(
        r"\[!\[([^\]\n]*)\]\((?:<[^>]+>|[^)\s]+)\)\]"
        r"\((?:<([^>]+)>|([^\s)]+))\)"
    )
    return [
        (match.start(), match.end(), match.group(1), match.group(2) or match.group(3))
        for match in pattern.finditer(content)
    ]


def extract_markdown_links(md_file: Path) -> list[LinkReference]:
    """Extract inline, reference-definition, autolink, and bare HTTP links."""
    content = md_file.read_text(encoding="utf-8")
    raw: list[tuple[int, int, str, str]] = _inline_links(content)
    raw.extend(_outer_image_links(content))

    reference_pattern = re.compile(
        r"(?m)^\s*\[([^\]]+)\]:\s*(?:<([^>]+)>|(\S+))"
    )
    for match in reference_pattern.finditer(content):
        raw.append(
            (match.start(), match.end(), match.group(1), match.group(2) or match.group(3))
        )

    occupied = [(start, end) for start, end, _, _ in raw]
    bare_pattern = re.compile(r"https?://[^\s<>\"']+")
    for match in bare_pattern.finditer(content):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        target = _trim_bare_url(match.group(0))
        raw.append((match.start(), match.start() + len(target), target, target))

    seen: set[tuple[int, str]] = set()
    references = []
    for start, _, label, target in sorted(raw):
        line = content.count("\n", 0, start) + 1
        key = (line, target)
        if key in seen:
            continue
        seen.add(key)
        references.append(LinkReference(md_file, line, label.strip(), target.strip()))
    return references


def extract_links(filepath: str | Path) -> list[str]:
    """Backward-compatible helper returning external HTTP(S) targets."""
    return [
        reference.target
        for reference in extract_markdown_links(Path(filepath))
        if reference.target.startswith(("http://", "https://"))
    ]


def github_heading_anchors(md_file: Path) -> set[str]:
    """Approximate GitHub's deterministic heading-slug algorithm."""
    anchors: set[str] = set()
    occurrences: Counter[str] = Counter()
    in_fence = False
    for line in md_file.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        heading = re.sub(r"<[^>]+>", "", match.group(1))
        heading = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", heading)
        heading = heading.casefold().strip()
        heading = re.sub(r"[^\w\- ]", "", heading, flags=re.UNICODE)
        slug = re.sub(r"\s", "-", heading)
        suffix = occurrences[slug]
        occurrences[slug] += 1
        anchors.add(slug if suffix == 0 else f"{slug}-{suffix}")
    return anchors


def _offline_error(reference: LinkReference, repo_root: Path) -> str | None:
    target = reference.target
    if not target or target.startswith(("mailto:", "tel:", "data:")):
        return None
    lowered = target.casefold()
    if any(marker in lowered for marker in PLACEHOLDER_MARKERS):
        return "placeholder URL"

    parsed = urllib.parse.urlsplit(target)
    if parsed.scheme in {"http", "https"}:
        if not parsed.netloc or any(character.isspace() for character in target):
            return "malformed external URL"
        return None
    if parsed.scheme:
        return f"unsupported URL scheme {parsed.scheme!r}"

    raw_path = urllib.parse.unquote(parsed.path)
    if raw_path.startswith("/"):
        return "repository-local path must be relative"
    candidate = reference.source if not raw_path else reference.source.parent / raw_path
    try:
        candidate = candidate.resolve(strict=False)
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        return "relative path escapes repository root"
    if not candidate.exists():
        return f"missing local target {raw_path!r}"
    if parsed.fragment:
        if candidate.is_dir():
            candidate = candidate / "README.md"
        if candidate.suffix.casefold() != ".md" or not candidate.exists():
            return "fragment target is not a Markdown document"
        fragment = urllib.parse.unquote(parsed.fragment).casefold()
        if fragment not in github_heading_anchors(candidate):
            return f"missing Markdown anchor #{fragment}"
    return None


def audit_offline(
    repo_root: Path,
    references: Iterable[LinkReference],
) -> list[tuple[LinkReference, str]]:
    """Return conclusive offline errors for the supplied references."""
    failures = []
    for reference in references:
        error = _offline_error(reference, repo_root)
        if error:
            failures.append((reference, error))
    return failures


def _classify_http_error(error: urllib.error.HTTPError, attempts: int) -> LinkResult:
    status = error.code
    if status in BROKEN_STATUS:
        category = "broken"
    elif status in BLOCKED_STATUS:
        category = "blocked"
    elif status in TRANSIENT_STATUS or status >= 500:
        category = "transient"
    else:
        category = "uncertain"
    error_url = getattr(error, "url", None) or getattr(error, "filename", None)
    return LinkResult(
        str(error_url or "<unknown>"),
        category,
        status,
        str(error.reason),
        attempts,
    )


def _looks_like_soft_404(final_url: str, sample: bytes) -> str | None:
    """Return evidence for common HTML soft-404 responses."""
    path = urllib.parse.urlsplit(final_url).path.casefold()
    if re.search(r"(?:^|[/_-])(?:404|not[-_]?found)(?:$|[./_-])", path):
        return f"redirected to soft-404 path {path!r}"
    text = sample.decode("utf-8", errors="ignore")
    title = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.I | re.S)
    if title and re.search(r"\b(?:404|not found|page unavailable)\b", title.group(1), re.I):
        compact = re.sub(r"\s+", " ", title.group(1)).strip()
        return f"soft-404 HTML title {compact!r}"
    return None


def check_url(
    url: str,
    timeout: float = 15.0,
    retries: int = 2,
    opener: Callable[..., object] = urllib.request.urlopen,
    sleeper: Callable[[float], None] = time.sleep,
) -> LinkResult:
    """Check one URL with HEAD, bounded retry, and a conservative GET fallback."""
    if timeout <= 0 or retries < 0:
        raise ValueError("timeout must be positive and retries non-negative")
    attempts = 0
    last_result: LinkResult | None = None
    for retry in range(retries + 1):
        attempts += 1
        for method in ("HEAD", "GET"):
            headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
            if method == "GET":
                headers["Range"] = "bytes=0-8191"
            request = urllib.request.Request(url, method=method, headers=headers)
            try:
                with opener(request, timeout=timeout) as response:
                    status = getattr(response, "status", 200)
                    final_url = getattr(response, "url", url)
                    sample = b""
                    if method == "GET" and hasattr(response, "read"):
                        sample = response.read(8192)
                if 200 <= status < 400:
                    soft_404 = _looks_like_soft_404(final_url, sample)
                    if soft_404:
                        return LinkResult(
                            url, "broken", status, soft_404, attempts, final_url
                        )
                    if method == "HEAD":
                        # A bounded GET is required to identify HTML soft-404s.
                        continue
                    return LinkResult(url, "ok", status, method, attempts, final_url)
                last_result = LinkResult(
                    url, "uncertain", status, f"unexpected {method} status", attempts
                )
            except urllib.error.HTTPError as error:
                result = _classify_http_error(error, attempts)
                result = dataclasses.replace(result, url=url)
                # GET can succeed where HEAD is unsupported or blocked.
                if method == "HEAD" and error.code in {403, 405, 501}:
                    last_result = result
                    continue
                last_result = result
            except (urllib.error.URLError, TimeoutError, ssl.SSLError, OSError) as error:
                last_result = LinkResult(
                    url, "transport", None, f"{type(error).__name__}: {error}", attempts
                )
            break

        if last_result and last_result.category == "broken":
            return last_result
        if retry < retries:
            sleeper(min(0.25 * (2**retry), 2.0))
    assert last_result is not None
    return last_result


def check_link(url: str, timeout: float = 10.0) -> bool:
    """Backward-compatible boolean helper; uncertain results are not healthy."""
    return check_url(url, timeout=timeout).healthy


def check_external_urls(
    urls: Iterable[str],
    timeout: float,
    retries: int,
    workers: int,
) -> list[LinkResult]:
    """Check unique URLs concurrently and return results in URL order."""
    unique_urls = sorted(set(urls))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            url: executor.submit(check_url, url, timeout, retries) for url in unique_urls
        }
        return [futures[url].result() for url in unique_urls]


def _json_report(
    repo_root: Path,
    references: list[LinkReference],
    offline_failures: list[tuple[LinkReference, str]],
    online_results: list[LinkResult],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "markdown_files": len(iter_markdown_files(repo_root)),
        "link_references": len(references),
        "unique_external_urls": len(
            {
                reference.target
                for reference in references
                if reference.target.startswith(("http://", "https://"))
            }
        ),
        "offline_failures": [
            {
                "file": str(reference.source.relative_to(repo_root)),
                "line": reference.line,
                "target": reference.target,
                "detail": detail,
            }
            for reference, detail in offline_failures
        ],
        "online_results": [dataclasses.asdict(result) for result in online_results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--offline", action="store_true", help="skip all network requests")
    parser.add_argument("--fail-on-uncertain", action="store_true")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 32:
        parser.error("--workers must be between 1 and 32")

    repo_root = args.repo_root.resolve()
    md_files = iter_markdown_files(repo_root)
    references = [
        reference
        for md_file in md_files
        for reference in extract_markdown_links(md_file)
    ]
    offline_failures = audit_offline(repo_root, references)
    online_results: list[LinkResult] = []
    if not args.offline:
        online_results = check_external_urls(
            (
                reference.target
                for reference in references
                if reference.target.startswith(("http://", "https://"))
            ),
            args.timeout,
            args.retries,
            args.workers,
        )

    report = _json_report(repo_root, references, offline_failures, online_results)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    for reference, detail in offline_failures:
        location = reference.source.relative_to(repo_root)
        print(f"OFFLINE {location}:{reference.line}: {detail}: {reference.target}")
    for result in online_results:
        if not result.healthy:
            status = result.status if result.status is not None else "network"
            print(f"ONLINE {result.category.upper()} {status}: {result.url} ({result.detail})")

    counts = Counter(result.category for result in online_results)
    print(
        f"Audited {len(references)} references in {len(md_files)} Markdown files; "
        f"offline failures={len(offline_failures)}, online={dict(sorted(counts.items()))}."
    )
    if offline_failures or any(result.category == "broken" for result in online_results):
        return 2
    uncertain = {"blocked", "transient", "transport", "uncertain"}
    if args.fail_on_uncertain and any(result.category in uncertain for result in online_results):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

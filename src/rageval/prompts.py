"""Helpers for loading markdown prompt files and extracting their frontmatter."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    meta_block, body = m.group(1), m.group(2)
    meta: dict[str, str] = {}
    for line in meta_block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, body


@lru_cache(maxsize=16)
def load_prompt(path: str) -> tuple[dict[str, str], str]:
    """Load a markdown prompt, returning (frontmatter, body)."""
    text = Path(path).read_text(encoding="utf-8")
    return _parse_frontmatter(text)


def render(body: str, variables: dict[str, Any]) -> str:
    """Simple str.format-based templating with safe missing-key handling.

    Variables can use `{name}` placeholders. Uses `str.format_map` with a
    defaulting dict so stray braces in the template (eg. JSON examples
    escaped with `{{` / `}}`) are preserved.
    """

    class _Defaulting(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return body.format_map(_Defaulting(variables))
